use pyo3::prelude::*;
use pyo3::types::PyDict;
use regex::Regex;
use std::collections::{HashMap, BTreeMap, BinaryHeap, HashSet};
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Regex helpers (compiled once, cached globally)
// ---------------------------------------------------------------------------

fn pre_tokenize_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d|\w+|[^\s\w]+").unwrap()
    })
}

fn special_marker_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"<think>|</think>|<search>|</search>|Q:|A:|User:|Bot:|Human:|Assistant:").unwrap()
    })
}

/// Check if char before `pos` is NOT ASCII-alpha (simulates lookbehind).
#[inline(always)]
fn is_non_alpha_boundary(text: &str, pos: usize) -> bool {
    pos == 0 || !text.as_bytes()[pos - 1].is_ascii_alphabetic()
}

/// Pre-tokenize text into words, matching the Python implementation.
fn pre_tokenize(text: &str) -> Vec<String> {
    let mut result = Vec::new();
    let special_re = special_marker_re();
    let word_re = pre_tokenize_re();

    let mut last_end = 0;
    for mat in special_re.find_iter(text) {
        let marker = mat.as_str();
        let needs_boundary = !marker.starts_with('<');
        if needs_boundary && !is_non_alpha_boundary(text, mat.start()) {
            continue;
        }
        if mat.start() > last_end {
            let segment = &text[last_end..mat.start()];
            for m in word_re.find_iter(segment) {
                let w = m.as_str().trim();
                if !w.is_empty() {
                    result.push(w.to_string());
                }
            }
        }
        match marker {
            "<think>" | "</think>" | "<search>" | "</search>" => result.push(marker.to_string()),
            "Q:" => result.push("<Q>".to_string()),
            "A:" => result.push("<A>".to_string()),
            "User:" | "Human:" => result.push("<USER>".to_string()),
            "Bot:" | "Assistant:" => result.push("<BOT>".to_string()),
            _ => {}
        }
        last_end = mat.end();
    }
    if last_end < text.len() {
        let segment = &text[last_end..];
        for m in word_re.find_iter(segment) {
            let w = m.as_str().trim();
            if !w.is_empty() {
                result.push(w.to_string());
            }
        }
    }
    result
}

// ---------------------------------------------------------------------------
// v2 pre-tokenizer (leading-space merges, per-digit) — port of
// enigma_engine/core/pretokenize.py. PARITY WITH PYTHON IS THE CONTRACT:
// every class below is spelled out to match PYTHON's `\w`/`\d`/`\s`
// semantics, NOT rust's `\w` (which adds marks/join-controls python lacks):
//   python \w = [\p{L}\p{Nd}\p{Nl}\p{No}_]      (str.isalnum() + underscore)
//   python \d = \p{Nd}
//   python \s = [\p{White_Space}\x1C-\x1F]      (FS/GS/RS/US are py-space)
//
// CONTRACT BOUNDARY (2026-07-20 audit): parity is exhaustive for every
// codepoint ASSIGNED as of Unicode 15.0 (CPython 3.12's tables; all
// 0x110000 scanned, 0 mismatches). Codepoints newer than 15.0 (~5k:
// Egyptian Hieroglyphs Ext-A, CJK Ext I, ...) are `Cn` to python but
// letters here (regex crate ships 16.0), so ids can differ when one
// abuts a contraction — different-but-lossless splits, rust arguably
// more correct. Consequence: v2 serving/training must encode through
// THIS backend, never the python fallback, whenever both exist.
// ---------------------------------------------------------------------------

/// Python's V2_PATTERN minus its `\s+(?!\S)` branch (rust `regex` has no
/// lookahead); `split_v2_segment` reproduces that branch's behaviour by
/// giving back the final whitespace char of a run followed by non-space.
/// Branch ORDER matters (leftmost-first alternation, same as python re).
fn v2_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(concat!(
            r"(?i:'(?:[sdmt]|ll|ve|re))",                       // contractions
            r"| ?[\p{L}\p{Nl}\p{No}]+",                         // ` ?[^\W\d_]+`
            r"| ?\p{Nd}",                                       // ` ?\d` (ONE digit)
            r"| ?(?:[^\s\x1C-\x1F\p{L}\p{Nd}\p{Nl}\p{No}_]|_)+", // ` ?(?:[^\s\w]|_)+`
            r"|[\s\x1C-\x1F]+",                                 // whitespace runs
        ))
        .unwrap()
    })
}

/// Python `\s` membership (rust is_whitespace = White_Space property,
/// which misses the \x1C-\x1F separators python counts as space).
#[inline(always)]
fn is_py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1C}'..='\u{1F}').contains(&c)
}

/// Split one carved segment with the v2 pattern.
///
/// Lookahead emulation: python's `\s+(?!\S)` branch makes a whitespace run
/// followed by a word surrender its LAST char, so a literal space can join
/// the next ` ?...` branch (" cat") while a tab stands alone. Here the
/// plain `\s+` branch matches the whole run and, when non-whitespace
/// follows, the run's last char is given back and re-matched.
fn split_v2_segment(text: &str, out: &mut Vec<String>) {
    let re = v2_re();
    let mut pos = 0usize;
    while pos < text.len() {
        let m = match re.find(&text[pos..]) {
            Some(m) => m,
            None => break, // unreachable: the branches cover every char
        };
        // python findall silently skips unmatched text; mirror that
        let start = pos + m.start();
        let end = pos + m.end();
        let piece = &text[start..end];
        if end < text.len() && piece.chars().all(is_py_space) {
            let n_chars = piece.chars().count();
            if n_chars > 1 {
                let (last_off, _) = piece.char_indices().last().unwrap();
                out.push(piece[..last_off].to_string());
                pos = start + last_off; // re-match from the surrendered char
                continue;
            }
            // single whitespace char before non-space: stands alone
            // (python's final `\s+` branch), fall through
        }
        out.push(piece.to_string());
        pos = end;
    }
}

// ---------------------------------------------------------------------------
// Symbol interning — maps strings to dense u32 IDs for allocation-free merges
// ---------------------------------------------------------------------------

struct SymbolTable {
    str_to_sym: HashMap<String, u32>,
    sym_to_str: Vec<String>,
}

impl SymbolTable {
    fn new() -> Self {
        SymbolTable {
            str_to_sym: HashMap::new(),
            sym_to_str: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.str_to_sym.clear();
        self.sym_to_str.clear();
    }

    /// Intern a string, returning its u32 symbol ID.
    fn intern(&mut self, s: &str) -> u32 {
        if let Some(&id) = self.str_to_sym.get(s) {
            return id;
        }
        let id = self.sym_to_str.len() as u32;
        self.sym_to_str.push(s.to_string());
        self.str_to_sym.insert(s.to_string(), id);
        id
    }

    /// Intern a string we already own (avoids one clone).
    fn intern_owned(&mut self, s: String) -> u32 {
        if let Some(&id) = self.str_to_sym.get(&s) {
            return id;
        }
        let id = self.sym_to_str.len() as u32;
        self.str_to_sym.insert(s.clone(), id);
        self.sym_to_str.push(s);
        id
    }

    #[inline(always)]
    fn resolve(&self, id: u32) -> &str {
        &self.sym_to_str[id as usize]
    }
}

// ---------------------------------------------------------------------------
// BPE merge using interned symbols — zero allocation in the hot loop
// ---------------------------------------------------------------------------

/// Merge table entry: (sym_a, sym_b) → (merged_sym, rank)
struct MergeEntry {
    merged: u32,
    rank: usize,
}

/// BPE merge on a word represented as interned symbol IDs.
/// Uses a linked-list-style skip array for O(1) removal.
fn merge_word(
    syms: &mut Vec<u32>,
    merge_table: &HashMap<u64, MergeEntry>,
) {
    if syms.len() <= 1 {
        return;
    }

    // next[i] = index of next valid symbol after i (skip array)
    let mut next: Vec<usize> = (1..syms.len() + 1).collect();
    // prev[i] = index of previous valid symbol
    let mut prev: Vec<usize> = Vec::with_capacity(syms.len());
    prev.push(usize::MAX); // sentinel for first
    for i in 1..syms.len() {
        prev.push(i - 1);
    }
    let len = syms.len();

    loop {
        // Find the lowest-rank merge among adjacent pairs
        let mut best_rank = usize::MAX;
        let mut best_i = usize::MAX;

        let mut i = 0;
        while next[i] < len {
            let j = next[i];
            let key = pair_key(syms[i], syms[j]);
            if let Some(entry) = merge_table.get(&key) {
                if entry.rank < best_rank {
                    best_rank = entry.rank;
                    best_i = i;
                }
            }
            i = j;
        }

        if best_i == usize::MAX {
            break;
        }

        // Merge: replace syms[best_i] with merged, remove syms[next[best_i]]
        let j = next[best_i];
        let key = pair_key(syms[best_i], syms[j]);
        let merged_sym = merge_table[&key].merged;
        syms[best_i] = merged_sym;

        // Skip j
        let after_j = next[j];
        next[best_i] = after_j;
        if after_j < len {
            prev[after_j] = best_i;
        }
    }

    // Compact: collect remaining symbols via the skip chain
    let mut compacted = Vec::with_capacity(16);
    let mut i = 0;
    loop {
        compacted.push(syms[i]);
        if next[i] >= len {
            break;
        }
        i = next[i];
    }
    *syms = compacted;
}

/// Pack two u32 symbol IDs into a single u64 key for HashMap lookup.
#[inline(always)]
fn pair_key(a: u32, b: u32) -> u64 {
    ((a as u64) << 32) | (b as u64)
}

/// Unpack a u64 pair key back to two u32 symbol IDs.
#[inline(always)]
fn unpack_pair_key(key: u64) -> (u32, u32) {
    ((key >> 32) as u32, key as u32)
}

// ---------------------------------------------------------------------------
// RustBPETokenizer — PyO3 class
// ---------------------------------------------------------------------------

#[pyclass]
struct RustBPETokenizer {
    // String-level maps (for save/load/decode/get_vocab)
    token_to_id: HashMap<String, i64>,
    id_to_token: HashMap<i64, String>,
    merges: Vec<(String, String)>,
    special_tokens: HashMap<String, i64>,
    use_utf8_bytes: bool,
    vocab_size: usize,

    // Interned symbol table + merge table (for fast encode)
    symbols: SymbolTable,
    merge_table: HashMap<u64, MergeEntry>,    // pair_key(a,b) → MergeEntry
    special_sym_ids: HashMap<u32, i64>,        // symbol_id → token_id for specials
    sym_to_token_id: HashMap<u32, i64>,        // symbol_id → token_id (all vocab)
    eow_sym: u32,                              // interned "</w>"

    // Char → symbol ID cache (for initial word decomposition)
    char_sym_cache: HashMap<char, u32>,

    // Convenience IDs
    pad_token_id: i64,
    bos_token_id: i64,
    eos_token_id: i64,
    unk_token_id: i64,

    // Cache: word string → final token IDs (skip merge + lookup on hit)
    cache: HashMap<String, Vec<i64>>,
    cache_order: Vec<String>,
    cache_cap: usize,

    // Pre-tokenizer version ("v1" | "v2"), read from the vocab file on
    // load (absence = v1, mirroring normalize_pretokenizer).
    pretokenizer: String,
    // v2 only: multi-char special literals, longest-first, as one
    // alternation — the carve-out of pretokenize_v2_with_specials.
    carve_re: Option<Regex>,
}

#[pymethods]
impl RustBPETokenizer {
    #[new]
    fn new() -> Self {
        let mut symbols = SymbolTable::new();
        let eow_sym = symbols.intern("</w>");
        RustBPETokenizer {
            token_to_id: HashMap::new(),
            id_to_token: HashMap::new(),
            merges: Vec::new(),
            special_tokens: HashMap::new(),
            use_utf8_bytes: false,
            vocab_size: 0,
            symbols,
            merge_table: HashMap::new(),
            special_sym_ids: HashMap::new(),
            sym_to_token_id: HashMap::new(),
            eow_sym,
            char_sym_cache: HashMap::new(),
            pad_token_id: 0,
            bos_token_id: 1,
            eos_token_id: 2,
            unk_token_id: 3,
            cache: HashMap::new(),
            cache_order: Vec::new(),
            cache_cap: 40000,
            pretokenizer: "v1".to_string(),
            carve_re: None,
        }
    }

    /// Load tokenizer from a JSON file (same format as Python BPETokenizer.save()).
    fn load(&mut self, path: &str) -> PyResult<()> {
        let contents = std::fs::read_to_string(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Cannot read {}: {}", path, e)))?;
        let data: serde_json::Value = serde_json::from_str(&contents)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid JSON: {}", e)))?;

        // --- token_to_id ---
        self.token_to_id.clear();
        if let Some(obj) = data.get("token_to_id").and_then(|v| v.as_object()) {
            for (k, v) in obj {
                if let Some(id) = v.as_i64() {
                    self.token_to_id.insert(k.clone(), id);
                }
            }
        } else {
            return Err(pyo3::exceptions::PyKeyError::new_err("Missing 'token_to_id'"));
        }

        // --- id_to_token ---
        self.id_to_token.clear();
        for (tok, &id) in &self.token_to_id {
            self.id_to_token.insert(id, tok.clone());
        }

        // --- merges ---
        self.merges.clear();
        if let Some(arr) = data.get("merges").and_then(|v| v.as_array()) {
            for m in arr {
                if let Some(pair) = m.as_array() {
                    if pair.len() == 2 {
                        let a = pair[0].as_str().unwrap_or("").to_string();
                        let b = pair[1].as_str().unwrap_or("").to_string();
                        self.merges.push((a, b));
                    }
                }
            }
        }

        // --- special_tokens ---
        self.special_tokens.clear();
        if let Some(obj) = data.get("special_tokens").and_then(|v| v.as_object()) {
            for (k, v) in obj {
                if let Some(id) = v.as_i64() {
                    self.special_tokens.insert(k.clone(), id);
                }
            }
        }

        // --- use_utf8_bytes ---
        self.use_utf8_bytes = data.get("use_utf8_bytes")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // --- pretokenizer version (absent/empty = v1; unknown = loud
        // failure, mirroring normalize_pretokenizer) ---
        self.pretokenizer = match data.get("pretokenizer").and_then(|v| v.as_str()) {
            None | Some("") | Some("v1") => "v1".to_string(),
            Some("v2") => "v2".to_string(),
            Some(other) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown pretokenizer version {:?}; expected one of ['v1', 'v2']",
                    other
                )))
            }
        };

        // --- v2 carve-out: every multi-char special literal, longest
        // first (char length, like python's key=len). Equal-length
        // distinct literals cannot compete for the same match position,
        // so the tie order is irrelevant.
        self.carve_re = if self.pretokenizer == "v2" {
            let mut tags: Vec<&String> = self
                .special_tokens
                .keys()
                .filter(|t| t.chars().count() > 1)
                .collect();
            if tags.is_empty() {
                None
            } else {
                tags.sort_by_key(|t| std::cmp::Reverse(t.chars().count()));
                let alt = tags
                    .iter()
                    .map(|t| regex::escape(t))
                    .collect::<Vec<_>>()
                    .join("|");
                Some(Regex::new(&alt).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "carve pattern failed to compile: {}",
                        e
                    ))
                })?)
            }
        } else {
            None
        };

        // --- Build interned tables from loaded data ---
        self.symbols.clear();
        self.eow_sym = self.symbols.intern("</w>");
        self.rebuild_encode_tables();

        Ok(())
    }

    /// Save tokenizer to JSON file (same format as Python).
    fn save(&self, path: &str) -> PyResult<()> {
        let mut sorted_tokens: Vec<(&String, &i64)> = self.token_to_id.iter().collect();
        sorted_tokens.sort_by_key(|(_, id)| **id);
        let token_map: BTreeMap<&String, &i64> = sorted_tokens.into_iter().collect();

        let merges_json: Vec<Vec<&str>> = self.merges
            .iter()
            .map(|(a, b)| vec![a.as_str(), b.as_str()])
            .collect();

        let mut data = serde_json::json!({
            "token_to_id": token_map,
            "merges": merges_json,
            "special_tokens": self.special_tokens,
            "use_utf8_bytes": self.use_utf8_bytes,
        });
        // v1 must NOT write the key (absence = v1 is the compat contract);
        // dropping it from a v2 save would silently v1-ify the vocab.
        if self.pretokenizer != "v1" {
            data["pretokenizer"] = serde_json::json!(self.pretokenizer);
        }

        let json_str = serde_json::to_string_pretty(&data)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("JSON serialization failed: {}", e)))?;

        let tmp_path = format!("{}.tmp", path);
        std::fs::write(&tmp_path, &json_str)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Cannot write {}: {}", tmp_path, e)))?;
        std::fs::rename(&tmp_path, path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Cannot rename to {}: {}", path, e)))?;
        Ok(())
    }

    /// Encode text to token IDs (matches Python BPETokenizer.encode()).
    #[pyo3(signature = (text, add_special_tokens=true))]
    fn encode(&mut self, text: &str, add_special_tokens: bool) -> Vec<i64> {
        let words = if self.pretokenizer == "v2" {
            self.pre_tokenize_v2_pieces(text)
        } else {
            pre_tokenize(text)
        };
        // Estimate output size: ~1.3 tokens per word + 2 special tokens
        let mut ids = Vec::with_capacity(words.len() * 2 + 2);

        if add_special_tokens {
            ids.push(self.bos_token_id);
        }

        for word in &words {
            if word.is_empty() {
                continue;
            }

            let actual_word: String = if self.use_utf8_bytes {
                word.as_bytes().iter().map(|&b| b as char).collect()
            } else {
                word.clone()
            };

            // Special token shortcut
            if let Some(&id) = self.special_tokens.get(&actual_word) {
                ids.push(id);
                continue;
            }

            // Check direct-ID cache (no merge + no lookup needed)
            if let Some(cached_ids) = self.cache.get(&actual_word) {
                ids.extend_from_slice(cached_ids);
                continue;
            }

            // Cache miss: run BPE merge with interned symbols
            let word_ids = self.tokenize_word_interned(&actual_word);

            // Insert into cache
            if self.cache.len() >= self.cache_cap {
                let evict_count = self.cache_cap / 4;
                let to_remove: Vec<String> = self.cache_order
                    .drain(..evict_count.min(self.cache_order.len()))
                    .collect();
                for key in to_remove {
                    self.cache.remove(&key);
                }
            }
            self.cache.insert(actual_word.clone(), word_ids.clone());
            self.cache_order.push(actual_word);

            ids.extend_from_slice(&word_ids);
        }

        if add_special_tokens {
            ids.push(self.eos_token_id);
        }

        ids
    }

    /// Decode token IDs back to text (matches Python BPETokenizer.decode()).
    #[pyo3(signature = (ids, skip_special_tokens=true))]
    fn decode(&self, ids: Vec<i64>, skip_special_tokens: bool) -> String {
        if self.pretokenizer == "v2" {
            // v2 decode is a plain lossless concatenation: NO </w>
            // substitution, NO space collapsing, NO trim (mirrors
            // _decode_v2 -- v1's whitespace surgery would corrupt v2
            // text, whose whitespace lives inside the pieces).
            let mut text = String::new();
            for id in ids {
                if let Some(token) = self.id_to_token.get(&id) {
                    if self.special_tokens.contains_key(token) {
                        if skip_special_tokens {
                            continue;
                        }
                        // literal tag; no "readable form" rewriting
                        text.push_str(token);
                        continue;
                    }
                    text.push_str(token);
                }
            }
            if self.use_utf8_bytes {
                let bytes: Vec<u8> = text.chars().map(|c| c as u8).collect();
                text = String::from_utf8_lossy(&bytes).to_string();
            }
            return text;
        }

        let mut tokens = Vec::with_capacity(ids.len());

        for id in ids {
            if let Some(token) = self.id_to_token.get(&id) {
                if self.special_tokens.contains_key(token) {
                    if skip_special_tokens {
                        continue;
                    }
                    let readable = match token.as_str() {
                        "<Q>" => "Q: ",
                        "<A>" => "A: ",
                        "<USER>" => "User: ",
                        "<BOT>" => "Bot: ",
                        _ => token.as_str(),
                    };
                    tokens.push(readable.to_string());
                    continue;
                }
                tokens.push(token.clone());
            }
        }

        let mut text = tokens.join("");
        text = text.replace("</w>", " ");

        static MULTI_SPACE: OnceLock<Regex> = OnceLock::new();
        let re = MULTI_SPACE.get_or_init(|| Regex::new(r" +").unwrap());
        text = re.replace_all(&text, " ").to_string();

        if self.use_utf8_bytes {
            let bytes: Vec<u8> = text.chars().map(|c| c as u8).collect();
            text = String::from_utf8_lossy(&bytes).to_string();
        }

        text.trim().to_string()
    }

    #[getter]
    fn get_vocab_size(&self) -> usize { self.vocab_size }
    #[getter]
    fn get_pad_token_id(&self) -> i64 { self.pad_token_id }
    #[getter]
    fn get_bos_token_id(&self) -> i64 { self.bos_token_id }
    #[getter]
    fn get_eos_token_id(&self) -> i64 { self.eos_token_id }
    #[getter]
    fn get_unk_token_id(&self) -> i64 { self.unk_token_id }
    #[getter]
    fn get_use_utf8_bytes(&self) -> bool { self.use_utf8_bytes }
    #[getter]
    fn get_pretokenizer(&self) -> String { self.pretokenizer.clone() }

    fn get_vocab(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        for (k, v) in &self.token_to_id {
            dict.set_item(k, v)?;
        }
        Ok(dict.into())
    }

    fn __len__(&self) -> usize { self.vocab_size }

    /// Train BPE tokenizer from text corpus.
    ///
    /// Returns a Python dict with `token_to_id`, `merges`, and
    /// `special_tokens` so the Python wrapper can sync its state.
    /// The Rust backend is also fully initialized for encode()/decode().
    #[pyo3(signature = (texts, vocab_size=32000, min_frequency=2, use_utf8_bytes=false, callback=None))]
    fn train(
        &mut self,
        py: Python<'_>,
        texts: Vec<String>,
        vocab_size: usize,
        min_frequency: usize,
        use_utf8_bytes: bool,
        callback: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        if texts.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Cannot train on empty text list",
            ));
        }

        // Phase 0: Reset to base vocabulary
        self.reset_to_base_vocab(use_utf8_bytes);
        let eow_sym = self.eow_sym;

        // Phase 1: Pre-tokenize and count word frequencies
        if let Some(ref cb) = callback {
            let _ = cb.call1(py, (0i32, "Pre-tokenizing...".to_string()));
        }

        let mut word_map: HashMap<Vec<u32>, u64> = HashMap::new();
        for text in &texts {
            let tokens = pre_tokenize(text);
            for word in tokens {
                let trimmed = word.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let actual: String = if use_utf8_bytes {
                    trimmed.as_bytes().iter().map(|&b| b as char).collect()
                } else {
                    trimmed.to_string()
                };
                let mut syms: Vec<u32> = actual
                    .chars()
                    .map(|c| self.symbols.intern(&c.to_string()))
                    .collect();
                syms.push(eow_sym);
                *word_map.entry(syms).or_insert(0) += 1;
            }
            // Yield GIL between texts for GUI responsiveness
            py.detach(|| {});
        }

        // Convert to indexed arrays
        let mut words: Vec<Vec<u32>> = Vec::with_capacity(word_map.len());
        let mut word_freq: Vec<u64> = Vec::with_capacity(word_map.len());
        for (syms, freq) in word_map {
            words.push(syms);
            word_freq.push(freq);
        }
        let unique_words = words.len();

        if let Some(ref cb) = callback {
            let msg = format!(
                "{} unique words, building pair map",
                unique_words
            );
            let _ = cb.call1(py, (0i32, msg));
        }

        // Phase 2: Build pair frequency map + reverse index
        let mut pair_freqs: HashMap<u64, u64> = HashMap::new();
        let mut pair_to_words: HashMap<u64, HashSet<usize>> = HashMap::new();

        for (widx, syms) in words.iter().enumerate() {
            let freq = word_freq[widx];
            for j in 0..syms.len().saturating_sub(1) {
                let pk = pair_key(syms[j], syms[j + 1]);
                *pair_freqs.entry(pk).or_insert(0) += freq;
                pair_to_words
                    .entry(pk)
                    .or_insert_with(HashSet::new)
                    .insert(widx);
            }
        }

        // Phase 3: Build max-heap (BinaryHeap is max by default)
        let mut heap: BinaryHeap<(u64, u64)> = BinaryHeap::new();
        for (&pk, &freq) in &pair_freqs {
            heap.push((freq, pk));
        }

        // Phase 4: Merge loop
        let base_size = self.token_to_id.len();
        let num_merges = vocab_size.saturating_sub(base_size);
        let min_freq = min_frequency as u64;

        if let Some(ref cb) = callback {
            let msg = format!(
                "Starting {} merges ({} unique words)",
                num_merges, unique_words
            );
            let _ = cb.call1(py, (0i32, msg));
        }

        let mut merge_pairs: Vec<(String, String)> = Vec::with_capacity(num_merges);

        for i in 0..num_merges {
            // Find best pair via lazy deletion from max-heap
            let mut best: Option<(u64, u64)> = None;
            while let Some((freq, pk)) = heap.pop() {
                let current = pair_freqs.get(&pk).copied().unwrap_or(0);
                if current > 0 && current == freq {
                    best = Some((pk, freq));
                    break;
                }
            }
            let (best_key, best_freq) = match best {
                Some(v) => v,
                None => break,
            };
            if best_freq < min_freq {
                break;
            }

            let (sym_a, sym_b) = unpack_pair_key(best_key);

            // Create merged token
            let a_str = self.symbols.resolve(sym_a).to_string();
            let b_str = self.symbols.resolve(sym_b).to_string();
            let merged_str = format!("{}{}", a_str, b_str);
            let merged_sym = self.symbols.intern(&merged_str);

            // Add to vocabulary
            if !self.token_to_id.contains_key(&merged_str) {
                let id = self.token_to_id.len() as i64;
                self.token_to_id.insert(merged_str.clone(), id);
                self.id_to_token.insert(id, merged_str);
            }

            // Record merge
            merge_pairs.push((a_str, b_str));

            // Incremental update: process only words containing merged pair
            let affected: Vec<usize> = pair_to_words
                .remove(&best_key)
                .unwrap_or_default()
                .into_iter()
                .collect();
            pair_freqs.remove(&best_key);

            for &widx in &affected {
                let freq = word_freq[widx];
                if freq == 0 {
                    continue;
                }

                // Clone old word to keep reference for pair removal
                let old_word = words[widx].clone();

                // Build new word with pair merged
                let mut new_word: Vec<u32> = Vec::with_capacity(old_word.len());
                let mut j = 0;
                while j < old_word.len() {
                    if j + 1 < old_word.len()
                        && old_word[j] == sym_a
                        && old_word[j + 1] == sym_b
                    {
                        new_word.push(merged_sym);
                        j += 2;
                    } else {
                        new_word.push(old_word[j]);
                        j += 1;
                    }
                }

                if new_word == old_word {
                    continue;
                }

                // Remove old pair contributions
                for j in 0..old_word.len().saturating_sub(1) {
                    let pk = pair_key(old_word[j], old_word[j + 1]);
                    if pk == best_key {
                        continue;
                    }
                    if let Some(f) = pair_freqs.get_mut(&pk) {
                        *f = f.saturating_sub(freq);
                        if *f == 0 {
                            pair_freqs.remove(&pk);
                        } else {
                            // S712: repush decremented pair so it stays
                            // visible in the heap
                            heap.push((*f, pk));
                        }
                    }
                    if let Some(set) = pair_to_words.get_mut(&pk) {
                        set.remove(&widx);
                        if set.is_empty() {
                            pair_to_words.remove(&pk);
                        }
                    }
                }

                // Add new pair contributions
                for j in 0..new_word.len().saturating_sub(1) {
                    let pk = pair_key(new_word[j], new_word[j + 1]);
                    let entry = pair_freqs.entry(pk).or_insert(0);
                    *entry += freq;
                    let current_freq = *entry;
                    pair_to_words
                        .entry(pk)
                        .or_insert_with(HashSet::new)
                        .insert(widx);
                    heap.push((current_freq, pk));
                }

                // Update word in place
                words[widx] = new_word;
            }

            // Yield GIL every 10 merges for GUI responsiveness
            if (i + 1) % 10 == 0 {
                py.detach(|| {});
            }
            // Progress callback every 100 merges
            if (i + 1) % 100 == 0 || i + 1 == num_merges {
                if let Some(ref cb) = callback {
                    let pct = ((i + 1) as u32 * 100) / num_merges as u32;
                    let msg = format!(
                        "Merge {}/{} (freq: {})",
                        i + 1,
                        num_merges,
                        best_freq
                    );
                    cb.call1(py, (pct as i32, msg))?;
                }
            }
        }

        // Final 100% callback
        if let Some(ref cb) = callback {
            cb.call1(py, (100i32, "Training complete".to_string()))?;
        }

        // Phase 5: Finalize — store merges and rebuild encode tables
        self.merges = merge_pairs;
        self.vocab_size = self.token_to_id.len();
        self.rebuild_encode_tables();

        // Build return dict for Python to sync its state
        let dict = PyDict::new(py);

        let tid_dict = PyDict::new(py);
        for (tok, &id) in &self.token_to_id {
            tid_dict.set_item(tok, id)?;
        }
        dict.set_item("token_to_id", tid_dict)?;

        let merges_list: Vec<Vec<&str>> = self.merges
            .iter()
            .map(|(a, b)| vec![a.as_str(), b.as_str()])
            .collect();
        dict.set_item("merges", merges_list)?;

        let st_dict = PyDict::new(py);
        for (tok, &id) in &self.special_tokens {
            st_dict.set_item(tok, id)?;
        }
        dict.set_item("special_tokens", st_dict)?;

        Ok(dict.into())
    }

    /// Return the merge list as a Python-friendly structure.
    fn get_merges(&self) -> Vec<Vec<String>> {
        self.merges
            .iter()
            .map(|(a, b)| vec![a.clone(), b.clone()])
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Private (non-PyO3) methods
// ---------------------------------------------------------------------------

impl RustBPETokenizer {
    /// Tokenize a word using interned BPE merges. Returns final token IDs.
    /// v2 pre-tokenize: carve special literals as whole units, split the
    /// text between them with the v2 pattern (pretokenize_v2_with_specials).
    fn pre_tokenize_v2_pieces(&self, text: &str) -> Vec<String> {
        let mut out = Vec::new();
        if text.is_empty() {
            return out;
        }
        match &self.carve_re {
            Some(cre) => {
                let mut last = 0usize;
                for m in cre.find_iter(text) {
                    if m.start() > last {
                        split_v2_segment(&text[last..m.start()], &mut out);
                    }
                    out.push(m.as_str().to_string());
                    last = m.end();
                }
                if last < text.len() {
                    split_v2_segment(&text[last..], &mut out);
                }
            }
            None => split_v2_segment(text, &mut out),
        }
        out
    }

    fn tokenize_word_interned(&mut self, word: &str) -> Vec<i64> {
        let is_v2 = self.pretokenizer == "v2";

        // Whole-piece vocab hit. v2 pieces carry no </w> marker (the
        // leading space IS the word boundary); v1 words end in </w>.
        if is_v2 {
            if let Some(&id) = self.token_to_id.get(word) {
                return vec![id];
            }
        } else {
            let word_with_end = format!("{}</w>", word);
            if let Some(&id) = self.token_to_id.get(&word_with_end) {
                return vec![id];
            }
        }

        // Decompose word into char symbols (+ </w> under v1 only)
        let mut syms: Vec<u32> = Vec::with_capacity(word.chars().count() + 1);
        for ch in word.chars() {
            let sym = if let Some(&cached) = self.char_sym_cache.get(&ch) {
                cached
            } else {
                let s = ch.to_string();
                let sym = self.symbols.intern(&s);
                self.char_sym_cache.insert(ch, sym);
                sym
            };
            syms.push(sym);
        }
        if !is_v2 {
            syms.push(self.eow_sym);
        }

        // Run BPE merges (zero-allocation inner loop)
        merge_word(&mut syms, &self.merge_table);

        // Convert merged symbols to token IDs
        let mut ids = Vec::with_capacity(syms.len());
        for &sym in &syms {
            if let Some(&token_id) = self.sym_to_token_id.get(&sym) {
                ids.push(token_id);
            } else {
                // Unknown symbol — try char-by-char
                let sym_str = self.symbols.resolve(sym);
                for ch in sym_str.chars() {
                    let ch_str = ch.to_string();
                    if let Some(&id) = self.token_to_id.get(&ch_str) {
                        ids.push(id);
                    } else {
                        ids.push(self.unk_token_id);
                    }
                }
            }
        }
        ids
    }
}

// ---------------------------------------------------------------------------
// BPE Training (R-2) — private helpers
// ---------------------------------------------------------------------------

/// Special tokens matching Python BPETokenizer.__init__ exactly.
/// Order + casing + coverage MUST stay identical to the Python dict
/// at enigma_engine/core/bpe_tokenizer.py L44 — after Rust train,
/// Python's self.special_tokens is replaced with this list verbatim
/// (see _try_rust_train), so any drift here clobbers Python-side
/// convenience IDs (think_start_id, search_start_id, etc.).
const SPECIAL_TOKENS: &[&str] = &[
    "<pad>", "<s>", "</s>", "<unk>", "<sep>",
    "<mask>", "<Q>", "<A>", "<USER>", "<BOT>",
    "<think>", "</think>", "<search>", "</search>",
];

impl RustBPETokenizer {
    /// Reset internal state and build base vocabulary (special tokens + 256
    /// byte-level tokens + `</w>`).  Matches Python `_init_base_vocab()`.
    fn reset_to_base_vocab(&mut self, use_utf8_bytes: bool) {
        self.token_to_id.clear();
        self.id_to_token.clear();
        self.merges.clear();
        self.special_tokens.clear();
        self.use_utf8_bytes = use_utf8_bytes;
        self.symbols.clear();
        self.eow_sym = self.symbols.intern("</w>");
        self.merge_table.clear();
        self.sym_to_token_id.clear();
        self.special_sym_ids.clear();
        self.char_sym_cache.clear();
        self.cache.clear();
        self.cache_order.clear();

        // Special tokens (IDs 0..13)
        for (i, &tok) in SPECIAL_TOKENS.iter().enumerate() {
            let id = i as i64;
            self.token_to_id.insert(tok.to_string(), id);
            self.id_to_token.insert(id, tok.to_string());
            self.special_tokens.insert(tok.to_string(), id);
        }

        // 256 byte-level tokens (skip any that collide with specials)
        let mut offset = SPECIAL_TOKENS.len() as i64;
        for byte_val in 0u16..=255 {
            let ch = char::from(byte_val as u8);
            let s = ch.to_string();
            if !self.token_to_id.contains_key(&s) {
                self.token_to_id.insert(s.clone(), offset);
                self.id_to_token.insert(offset, s);
                offset += 1;
            }
        }

        // </w> end-of-word marker
        if !self.token_to_id.contains_key("</w>") {
            let id = self.token_to_id.len() as i64;
            self.token_to_id.insert("</w>".to_string(), id);
            self.id_to_token.insert(id, "</w>".to_string());
        }

        self.vocab_size = self.token_to_id.len();
    }

    /// Rebuild all derived lookup tables (merge_table, sym_to_token_id, etc.)
    /// from the primary state (token_to_id, merges, special_tokens).
    /// Called after load() and train() to prepare for encode()/decode().
    fn rebuild_encode_tables(&mut self) {
        // Sync special IDs
        self.pad_token_id = *self.special_tokens.get("<pad>").unwrap_or(&0);
        self.bos_token_id = *self.special_tokens.get("<s>").unwrap_or(&1);
        self.eos_token_id = *self.special_tokens.get("</s>").unwrap_or(&2);
        self.unk_token_id = *self.special_tokens.get("<unk>").unwrap_or(&3);
        self.vocab_size = self.token_to_id.len();

        // Intern all vocabulary tokens
        self.sym_to_token_id.clear();
        for (tok, &id) in &self.token_to_id {
            let sym = self.symbols.intern_owned(tok.clone());
            self.sym_to_token_id.insert(sym, id);
        }

        // Intern special tokens
        self.special_sym_ids.clear();
        for (tok, &id) in &self.special_tokens {
            let sym = self.symbols.intern(tok);
            self.special_sym_ids.insert(sym, id);
        }

        // Build merge table
        self.merge_table.clear();
        for (rank, (a_str, b_str)) in self.merges.iter().enumerate() {
            let a_sym = self.symbols.intern(a_str);
            let b_sym = self.symbols.intern(b_str);
            let merged_str = format!("{}{}", a_str, b_str);
            let merged_sym = self.symbols.intern_owned(merged_str);
            self.merge_table.insert(
                pair_key(a_sym, b_sym),
                MergeEntry { merged: merged_sym, rank },
            );
        }

        // Pre-cache char → symbol mappings
        self.char_sym_cache.clear();
        for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789".chars() {
            let s = c.to_string();
            let sym = self.symbols.intern(&s);
            self.char_sym_cache.insert(c, sym);
        }

        // Clear encode cache
        self.cache.clear();
        self.cache_order.clear();
    }
}

// ---------------------------------------------------------------------------
// Python module
// ---------------------------------------------------------------------------

#[pymodule]
fn enigma_bpe(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustBPETokenizer>()?;
    // Feature flag for the Python side: a stale built extension without
    // this attribute must never be handed a v2 vocab (it would silently
    // split with v1 rules).
    m.add("V2_SUPPORTED", true)?;
    Ok(())
}
