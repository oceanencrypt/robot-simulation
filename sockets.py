# socket.py
import socketio
import logging
from simulation import RobotControlGUI, MujocoSimulator, run_simulation
import time
from threading import Event, Timer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class SocketClient:
    def __init__(self, server_url, simulator):
        self.sio = socketio.Client(
            reconnection=True,
            reconnection_attempts=0,
            reconnection_delay=1,
            reconnection_delay_max=5,
            logger=logger,
            request_timeout=1000
        )
        self.server_url = server_url
        self.simulator = simulator
        self.running = True
        self.connected = Event()
        self.last_message_time = time.time()
        self.message_queue = {}
        self.setup_handlers()
        self.start_heartbeat()

    def setup_handlers(self):
        self.sio.on("connect", self.on_connect)
        self.sio.on("disconnect", self.on_disconnect)
        self.sio.on("terafacMini", self.on_simulation_data)
        self.sio.on("connect_error", self.on_connect_error)
        self.sio.on("pong", self.on_pong)
        self.sio.on("test", self.test)
        self.sio.on("message_received", self.on_message_received)
        self.sio.on("connection_established", self.on_connection_established)

    def start_heartbeat(self):
        def send_heartbeat():
            while self.running:
                if self.sio.connected:
                    try:
                        self.sio.emit("ping")
                    except Exception as e:
                        logger.error(f"Error sending heartbeat: {str(e)}")
                time.sleep(15)  # Send heartbeat every 15 seconds

        self.heartbeat_thread = Timer(1, send_heartbeat)
        self.heartbeat_thread.daemon = True
        self.heartbeat_thread.start()

    def on_connect(self):
        logger.info("Connected to server")
        self.connected.set()
        self.last_message_time = time.time()

    def on_connection_established(self, data):
        logger.info(f"Connection established with client ID: {data['client_id']}")

    def on_disconnect(self):
        logger.info("Disconnected from server")
        self.connected.clear()
        if self.running:
            self.reconnect()

    def on_connect_error(self, error):
        logger.error(f"Connection error: {error}")
        self.connected.clear()
        if self.running:
            self.reconnect()

    def on_pong(self):
        self.last_message_time = time.time()
    
    def test(self, data):
        print("TTESTTT")

    def on_message_received(self, data):
        message_id = data.get('message_id')
        if message_id in self.message_queue:
            self.message_queue[message_id].set()

    def reconnect(self):
        """Attempt to reconnect to the server with exponential backoff"""
        retry_delay = 1
        max_delay = 30
        
        while self.running and not self.sio.connected:
            try:
                logger.info(f"Attempting to reconnect in {retry_delay} seconds...")
                time.sleep(retry_delay)
                self.connect()
                if self.sio.connected:
                    break
                
                retry_delay = min(retry_delay * 2, max_delay)
            except Exception as e:
                logger.error(f"Reconnection error: {str(e)}")

    def on_simulation_data(self, data):
        logger.info(f"Received simulation data: {data}")
        try:
            message_id = data.get('message_id')
            coordinates = data['coordinates']
            obj_url = data['obj_url']
            
            # Acknowledge receipt
            self.sio.emit("message", {
                "status": "simulation_started",
                "message_id": message_id
            })
            
            run_simulation(coordinates, obj_url, self.simulator)
            
            # Report completion
            self.sio.emit("message", {
                "status": "simulation_complete",
                "message_id": message_id
            })
            
        except Exception as e:
            logger.error(f"Simulation error: {str(e)}")
            if message_id:
                self.sio.emit("message", {
                    "status": "simulation_error",
                    "message_id": message_id,
                    "error": str(e)
                })

    def connect(self):
        try:
            if not self.sio.connected:
                self.sio.connect(
                    self.server_url,
                    transports=['websocket'],
                    wait_timeout=60
                )
            return True
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
            return False

    def run(self):
        while self.running:
            try:
                if not self.sio.connected:
                    self.connect()
                
                # Check connection health
                if self.sio.connected and time.time() - self.last_message_time > 60:
                    logger.warning("No messages received for 60 seconds, reconnecting...")
                    self.sio.disconnect()
                    continue
                
                self.sio.wait()
                time.sleep(0.1)
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                self.running = False
                self.heartbeat_thread.cancel()
                if self.sio.connected:
                    self.sio.disconnect()
                break
            except Exception as e:
                logger.error(f"Error: {str(e)}")
                if self.sio.connected:
                    self.sio.disconnect()
                time.sleep(1)

def main():
    server_url = "http://127.0.0.1:5000"
    simulator = MujocoSimulator()
    client = SocketClient(server_url, simulator)
    client.run()

if __name__ == "__main__":
    main()