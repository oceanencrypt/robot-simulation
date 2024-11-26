import logging
from flask import Flask, request
from flask_cors import CORS
from headless_simulation import MujocoSimulator, run_simulation


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

simulator = MujocoSimulator()


@app.route("/send_simulation_data", methods=["POST"])
def send_simulation_data():
    try:
        data = request.get_json()
        coordinates = data['coordinates']
        obj_url = data['obj_url']
        run_simulation(coordinates, obj_url, simulator)

        return {
            "status": "success",
        }, 200

    except Exception as e:
        logger.error(f"Error in send_simulation_data: {str(e)}")
        return {"status": "error", "message": str(e)}, 500


if __name__ == "__main__":
    logger.info("Starting server...")
    app.run()
