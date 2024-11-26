# server.py
import eventlet

eventlet.monkey_patch()
from flask import Flask, request
from flask_socketio import SocketIO
from flask_cors import CORS
import logging
from threading import Lock
import time  # Use regular time module instead

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Add thread synchronization
thread_lock = Lock()


# Add socket management
class SocketManager:
    def __init__(self):
        self.connected_clients = set()
        self.client_health = {}

    def add_client(self, client_id):
        with thread_lock:
            self.connected_clients.add(client_id)
            self.client_health[client_id] = {
                "last_active": time.time(),
                "message_count": 0,
            }

    def remove_client(self, client_id):
        with thread_lock:
            self.connected_clients.discard(client_id)
            self.client_health.pop(client_id, None)

    def update_client_activity(self, client_id):
        with thread_lock:
            if client_id in self.client_health:
                self.client_health[client_id]["last_active"] = time.time()
                self.client_health[client_id]["message_count"] += 1

    def get_active_clients(self):
        with thread_lock:
            current_time = time.time()
            return {
                cid
                for cid in self.connected_clients
                if current_time - self.client_health[cid]["last_active"] < 60
            }  # 60 second timeout


socket_manager = SocketManager()

sio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="eventlet",
    ping_timeout=1000,
    ping_interval=25,
    max_http_buffer_size=1e8,
    logger=True,  # Enable SocketIO logging
    engineio_logger=True,  # Enable Engine.IO logging
)


@sio.on("connect")
def connect():
    client_id = request.sid
    socket_manager.add_client(client_id)
    logger.info(f"Client connected. ID: {client_id}")
    # Send initial connection confirmation
    sio.emit("connection_established", {"client_id": client_id}, room=client_id)


@sio.on("disconnect")
def disconnect():
    client_id = request.sid
    socket_manager.remove_client(client_id)
    logger.info(f"Client disconnected. ID: {client_id}")


@sio.on("ping")
def handle_ping():
    client_id = request.sid
    socket_manager.update_client_activity(client_id)
    sio.emit("pong", room=client_id)


@sio.on("message")
def handle_message(data):
    client_id = request.sid
    socket_manager.update_client_activity(client_id)
    logger.info(f"Received message from {client_id}: {data}")
    # Add acknowledgment
    sio.emit(
        "message_received",
        {"status": "ok", "message_id": data.get("id")},
        room=client_id,
    )


@app.route("/send_simulation_data", methods=["POST"])
def send_simulation_data():
    try:
        active_clients = socket_manager.get_active_clients()
        if not active_clients:
            logger.warning("No active clients connected")
            return {"status": "error", "message": "No active clients connected"}, 503

        data = request.get_json()
        message_id = str(time.time())  # Generate unique message ID using regular time

        # Add message tracking
        data["message_id"] = message_id

        sio.emit('test', {})

        print('ACTIVE CLIENTS :::', active_clients)
        sio.emit('test', {})
        
        # Emit to all active clients
        for client_id in active_clients:
            try:
                sio.emit(
                    "test",
                    {},
                    room=client_id,
                    callback=lambda: logger.info(
                        f"Message {message_id} delivered to {client_id}"
                    ),
                )
                sio.emit(
                    "terafacMini",
                    data,
                    room=client_id,
                    callback=lambda: logger.info(
                        f"Message {message_id} delivered to {client_id}"
                    ),
                )
                socket_manager.update_client_activity(client_id)
            except Exception as e:
                logger.error(f"Failed to send to client {client_id}: {str(e)}")
                socket_manager.remove_client(client_id)

        return {
            "status": "success",
            "message_id": message_id,
            "clients_notified": len(active_clients),
        }, 200

    except Exception as e:
        logger.error(f"Error in send_simulation_data: {str(e)}")
        return {"status": "error", "message": str(e)}, 500


if __name__ == "__main__":
    logger.info("Starting server...")
    sio.run(app, host="127.0.0.1", port=5000, debug=True, use_reloader=False)
