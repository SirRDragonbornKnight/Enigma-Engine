#!/usr/bin/env python3
"""
Router - Central Hub for All Services

Manages connections from standalone services (imagegen, codegen, etc.)
and routes commands between them and clients.

Usage:
    python router.py                    # Start router on default port 9900
    python router.py --port 9900       # Custom port
    python router.py --list            # List connected services
"""

import argparse
import json
import logging
import socket
import struct
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Protocol
# =============================================================================


class MessageType(Enum):
    REGISTER = "register"
    COMMAND = "command"
    RESPONSE = "response"
    EVENT = "event"
    HEARTBEAT = "heartbeat"
    SHUTDOWN = "shutdown"
    LIST_SERVICES = "list_services"
    ROUTE = "route"


@dataclass
class Message:
    type: MessageType
    payload: Dict[str, Any] = field(default_factory=dict)
    id: str = ""

    def to_bytes(self) -> bytes:
        data = json.dumps({"type": self.type.value, "payload": self.payload, "id": self.id}).encode("utf-8")
        return struct.pack(">I", len(data)) + data

    @classmethod
    def from_bytes(cls, data: bytes) -> "Message":
        obj = json.loads(data.decode("utf-8"))
        return cls(type=MessageType(obj["type"]), payload=obj.get("payload", {}), id=obj.get("id", ""))


@dataclass
class ServiceInfo:
    """Information about a connected service."""

    name: str
    capabilities: List[str]
    commands: List[str]
    socket: socket.socket
    address: tuple
    connected_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)


# =============================================================================
# Router Server
# =============================================================================


class Router:
    """Central router for service coordination."""

    def __init__(self, port: int = 9900):
        self.port = port
        self.services: Dict[str, ServiceInfo] = {}
        self.capability_map: Dict[str, List[str]] = {}  # capability -> [service names]
        self._running = False
        self._server: Optional[socket.socket] = None
        self._lock = threading.Lock()
        self._pending_responses: Dict[str, threading.Event] = {}
        self._response_data: Dict[str, Dict] = {}
        self._clients: List[socket.socket] = []

    def start(self):
        """Start the router server."""
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("0.0.0.0", self.port))
        self._server.listen(10)

        self._running = True
        logger.info(f"Router listening on port {self.port}")

        # Start heartbeat thread
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()

        while self._running:
            try:
                self._server.settimeout(1.0)
                try:
                    client, addr = self._server.accept()
                except socket.timeout:
                    continue

                logger.info(f"Connection from {addr}")
                threading.Thread(target=self._handle_connection, args=(client, addr), daemon=True).start()

            except Exception as e:
                if self._running:
                    logger.error(f"Server error: {e}")

        self._server.close()

    def _handle_connection(self, conn: socket.socket, addr: tuple):
        """Handle a new connection (service or client)."""
        service_name = None

        try:
            while self._running:
                try:
                    len_data = conn.recv(4)
                    if not len_data:
                        break

                    msg_len = struct.unpack(">I", len_data)[0]
                    msg_data = b""
                    while len(msg_data) < msg_len:
                        chunk = conn.recv(min(4096, msg_len - len(msg_data)))
                        if not chunk:
                            break
                        msg_data += chunk

                    if len(msg_data) < msg_len:
                        break

                    msg = Message.from_bytes(msg_data)

                    if msg.type == MessageType.REGISTER:
                        # Service registration
                        service_name = self._register_service(msg.payload, conn, addr)
                        if service_name:
                            # Send acknowledgment
                            ack = Message(
                                type=MessageType.RESPONSE,
                                payload={"success": True, "message": f"Registered as {service_name}"},
                                id=msg.id,
                            )
                            conn.sendall(ack.to_bytes())

                    elif msg.type == MessageType.HEARTBEAT:
                        # Update heartbeat
                        if service_name and service_name in self.services:
                            self.services[service_name].last_heartbeat = time.time()

                    elif msg.type == MessageType.RESPONSE:
                        # Response from a service
                        if msg.id in self._pending_responses:
                            self._response_data[msg.id] = msg.payload
                            self._pending_responses[msg.id].set()

                    elif msg.type == MessageType.LIST_SERVICES:
                        # Client requesting service list
                        resp = Message(type=MessageType.RESPONSE, payload=self._get_services_info(), id=msg.id)
                        conn.sendall(resp.to_bytes())

                    elif msg.type == MessageType.ROUTE:
                        # Route command to a service
                        result = self._route_command(msg.payload)
                        resp = Message(type=MessageType.RESPONSE, payload=result, id=msg.id)
                        conn.sendall(resp.to_bytes())

                    elif msg.type == MessageType.COMMAND:
                        # Direct command (for clients)
                        target = msg.payload.get("service")
                        command = msg.payload.get("command")
                        params = msg.payload.get("params", {})

                        result = self._send_to_service(target, command, params)
                        resp = Message(type=MessageType.RESPONSE, payload=result, id=msg.id)
                        conn.sendall(resp.to_bytes())

                except socket.timeout:
                    continue
                except Exception as e:
                    logger.error(f"Connection error: {e}")
                    break

        except Exception as e:
            logger.error(f"Handler error: {e}")
        finally:
            if service_name:
                self._unregister_service(service_name)
            conn.close()

    def _register_service(self, payload: Dict[str, Any], conn: socket.socket, addr: tuple) -> Optional[str]:
        """Register a new service."""
        name = payload.get("name")
        if not name:
            return None

        with self._lock:
            # Check if service already registered
            if name in self.services:
                logger.warning(f"Service {name} already registered, replacing")
                old_conn = self.services[name].socket
                try:
                    old_conn.close()
                except Exception:
                    pass

            # Register service
            service = ServiceInfo(
                name=name,
                capabilities=payload.get("capabilities", []),
                commands=payload.get("commands", []),
                socket=conn,
                address=addr,
            )
            self.services[name] = service

            # Update capability map
            for cap in service.capabilities:
                if cap not in self.capability_map:
                    self.capability_map[cap] = []
                if name not in self.capability_map[cap]:
                    self.capability_map[cap].append(name)

            logger.info(f"Registered service: {name} with capabilities {service.capabilities}")
            return name

    def _unregister_service(self, name: str):
        """Unregister a service."""
        with self._lock:
            if name in self.services:
                service = self.services[name]

                # Remove from capability map
                for cap in service.capabilities:
                    if cap in self.capability_map:
                        if name in self.capability_map[cap]:
                            self.capability_map[cap].remove(name)

                del self.services[name]
                logger.info(f"Unregistered service: {name}")

    def _get_services_info(self) -> Dict[str, Any]:
        """Get information about connected services."""
        with self._lock:
            return {
                "success": True,
                "services": {
                    name: {
                        "capabilities": srv.capabilities,
                        "commands": srv.commands,
                        "connected_at": srv.connected_at,
                        "last_heartbeat": srv.last_heartbeat,
                    }
                    for name, srv in self.services.items()
                },
                "capabilities": dict(self.capability_map),
            }

    def _route_command(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Route a command based on capability or service name."""
        service = payload.get("service")
        capability = payload.get("capability")
        command = payload.get("command")
        params = payload.get("params", {})

        # Find target service
        target = None
        if service and service in self.services:
            target = service
        elif capability and capability in self.capability_map:
            services = self.capability_map[capability]
            if services:
                target = services[0]  # Use first available

        if not target:
            return {"success": False, "error": f"No service found for {service or capability}"}

        return self._send_to_service(target, command, params)

    def _send_to_service(self, service_name: str, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send a command to a specific service."""
        with self._lock:
            if service_name not in self.services:
                return {"success": False, "error": f"Service {service_name} not connected"}
            service = self.services[service_name]

        msg_id = str(uuid.uuid4())
        event = threading.Event()
        self._pending_responses[msg_id] = event

        try:
            msg = Message(type=MessageType.COMMAND, payload={"command": command, "params": params}, id=msg_id)
            service.socket.sendall(msg.to_bytes())

            # Wait for response with timeout
            if event.wait(timeout=60.0):
                result = self._response_data.pop(msg_id, {"success": False, "error": "No response data"})
            else:
                result = {"success": False, "error": "Timeout waiting for response"}

        except Exception as e:
            result = {"success": False, "error": str(e)}
        finally:
            self._pending_responses.pop(msg_id, None)

        return result

    def _heartbeat_loop(self):
        """Send heartbeats and check for dead services."""
        while self._running:
            time.sleep(30)

            now = time.time()
            dead_services = []

            with self._lock:
                for name, service in self.services.items():
                    # Send heartbeat
                    try:
                        msg = Message(type=MessageType.HEARTBEAT)
                        service.socket.sendall(msg.to_bytes())
                    except Exception:
                        dead_services.append(name)
                        continue

                    # Check if service is dead (no heartbeat for 2 minutes)
                    if now - service.last_heartbeat > 120:
                        dead_services.append(name)

            for name in dead_services:
                logger.warning(f"Service {name} appears dead, removing")
                self._unregister_service(name)

    def shutdown(self):
        """Shutdown the router."""
        logger.info("Shutting down router...")
        self._running = False

        # Send shutdown to all services
        with self._lock:
            for name, service in self.services.items():
                try:
                    msg = Message(type=MessageType.SHUTDOWN)
                    service.socket.sendall(msg.to_bytes())
                except Exception:
                    pass


# =============================================================================
# Client Helper
# =============================================================================


class RouterClient:
    """Client for connecting to the router."""

    def __init__(self, host: str = "localhost", port: int = 9900):
        self.host = host
        self.port = port
        self._socket: Optional[socket.socket] = None

    def connect(self) -> bool:
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.connect((self.host, self.port))
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    def disconnect(self):
        if self._socket:
            self._socket.close()
            self._socket = None

    def list_services(self) -> Dict[str, Any]:
        """Get list of connected services."""
        msg = Message(type=MessageType.LIST_SERVICES, id=str(uuid.uuid4()))
        return self._send_receive(msg)

    def send_command(self, service: str, command: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Send command to a specific service."""
        msg = Message(
            type=MessageType.COMMAND,
            payload={"service": service, "command": command, "params": params or {}},
            id=str(uuid.uuid4()),
        )
        return self._send_receive(msg)

    def route_by_capability(self, capability: str, command: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Route command by capability."""
        msg = Message(
            type=MessageType.ROUTE,
            payload={"capability": capability, "command": command, "params": params or {}},
            id=str(uuid.uuid4()),
        )
        return self._send_receive(msg)

    def _send_receive(self, msg: Message) -> Dict[str, Any]:
        if not self._socket:
            return {"success": False, "error": "Not connected"}

        try:
            self._socket.sendall(msg.to_bytes())

            len_data = self._socket.recv(4)
            if not len_data:
                return {"success": False, "error": "Connection closed"}

            msg_len = struct.unpack(">I", len_data)[0]
            msg_data = b""
            while len(msg_data) < msg_len:
                chunk = self._socket.recv(min(4096, msg_len - len(msg_data)))
                if not chunk:
                    break
                msg_data += chunk

            resp = Message.from_bytes(msg_data)
            return resp.payload
        except Exception as e:
            return {"success": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Router - Service Hub")
    parser.add_argument("--port", type=int, default=9900, help="Port to listen on")
    parser.add_argument("--list", action="store_true", help="List connected services")
    parser.add_argument("--host", type=str, default="localhost", help="Host for client mode")

    args = parser.parse_args()

    if args.list:
        # Client mode - list services
        client = RouterClient(args.host, args.port)
        if client.connect():
            result = client.list_services()
            print(json.dumps(result, indent=2, default=str))
            client.disconnect()
        return

    # Server mode
    router = Router(port=args.port)

    try:
        router.start()
    except KeyboardInterrupt:
        router.shutdown()


if __name__ == "__main__":
    main()
