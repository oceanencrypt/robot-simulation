import socketio
import json
import subprocess
import math
#import asyncio

PNC_SERVER = "https://level-1.feignbird.live/"
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJmcmVzaCI6ZmFsc2UsImlhdCI6MTczNzYyNTgxNiwianRpIjoiOTQ3MjNlYjMtMDQ2Ni00MmZkLTk0OGItMTY5MzU1NWYxMGE5IiwidHlwZSI6ImFjY2VzcyIsInN1YiI6ImR2bnNoMjMxMkBnbWFpbC5jb20iLCJuYmYiOjE3Mzc2MjU4MTYsImNzcmYiOiI0YWZjYWYwOS05Nzg5LTQxZTctODNmYS1lNTQ5NDVkMjcxZDUiLCJleHAiOjE3NjkxNjE4MTZ9.9AL8RnQFzjN7gc2Rag9_5VtuErZ7e9L8hxhjR6zl49o"
dummy_data = {'data': {'id': 49, 'project_id': 'b8956623-b9ea-4ef7-85e3-76f9bb8c1f0a', 'project_name': 'bigger-L', 'project_img': 'https://terafac-welding-pro.s3.amazonaws.com/welding_images/bigger-L.png', 'last_edited': '2024-10-07T12:42:24.240000Z', 'last_run': '2024-10-08T09:49:07.292000Z', 'run_time': '6', 'status': 'Completed', 'size': 0, 'welding_obj_file': 'https://terafac-welding-pro.s3.amazonaws.com/welding_objects/bigger-L.obj', 'welding_data': {'edges': [{'start': {'x': 1193, 'z': 24, 'y': -21.5}, 'end': {'x': 1307, 'z': 24, 'y': -21.5}, 'weld_length': 114, 'joint_type': 'Fillet', 'leg_length': 114, 'passes': [{'end_edge_offset': 2, 'id': 0, 'start_edge_offset': 2, 'travel_angle': 2, 'welding_speed': 48, 'work_angle': 2}], 'welding_position': 'Horizontal Position', 'segment_name': 'edge-10'}], 'no_of_edges': 23}, 'user_id': 5}}
# Create a SocketIO client instance to connect to the server
sio = socketio.Client()

@sio.event
def connect():
    print('Connected to the server')
    # token = login()
    sio.emit('receive_token', {'token': TOKEN})
    print(f"Token emitted to server: {TOKEN}")


@sio.event
def disconnect():
    print('Disconnected from the server')

@sio.on('status_update')
def handle_status_update(data):
    print(f"Received status update from server: {data}")

@sio.on('welding_status')
def handle_welding_status(data):
    print(f"Received welding status: {data}")

import numpy as np
@sio.on('welding_data')
def handle_welding_data(data):
    global MODE
    global welding_data
    print(f"\nReceived welding data")# : {data}

    # coordinates
    edges = data['data']['welding_data']['edges']

    coordinates = []
    #print(f"Edges are: {edges}")

    # for edge in edges:
    #     #Convert PnC x, y, z coordinates to simulation units
    #     start_x = edge['start']['x'] * 0.001 - 1.25 + 0.3
    #     end_x = edge['end']['x'] * 0.001 - 1.25 + 0.3
    #     start_y = edge['start']['y'] * 0.001 + 0.015
    #     end_y = edge['end']['y'] * 0.001
    #     start_z = edge['start']['z'] * 0.001 + 0.033 + 0.018
    #     end_z = edge['end']['z'] * 0.001 + 0.033 + 0.018

    #     if start_x < end_x:
    #         start_x += 0.0139
    #         end_x -= 0.0139
    #     elif start_x > end_x:
    #         start_x -= 0.0139
    #         end_x += 0.0139
    #     else:
    #         start_x = end_x
    for edge in edges:
    # Convert PnC x, y, z coordinates to simulation units
        start_x = edge['start']['x'] * 0.001 - 1.25 + 0.3
        end_x = edge['end']['x'] * 0.001 - 1.25 + 0.3

        # Apply adjustment when the start x-coordinate is greater than the end x-coordinate
        # if start_x > end_x:
             
        #      start_x -= 0.0139
        #      end_x += 0.0139
        start_y = edge['start']['y'] * 0.001
        start_z = edge['start']['z'] * 0.001

        end_y = edge['end']['y'] * 0.001
        end_z = edge['end']['z'] * 0.001



        coordinates.append((
            start_y,  # PnC y becomes sim x
            start_x, # Adjusted x-coordinate for sim y
            start_z  # PnC z becomes sim z
        ))

        coordinates.append((
            end_y,   # PnC y becomes sim x
            end_x,                     # Adjusted x-coordinate for sim y
            end_z   # PnC z becomes sim z
        ))
    edge_length = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2 + (end_z - start_z)**2)

    coordinates.append((start_y, start_x, start_z))
    coordinates.append((end_y, end_x, end_z))
    # Debugging print to verify the transformation
    print(f"Edge Start (PnC x, y, z): ({start_y}, {start_x}, {start_z})")
    print(f"Edge End   (PnC x, y, z): ({end_y}, {end_x}, {end_z})")
    print(f"edge length: {edge_length}")



       
    #print(f"Coordinates just after scaling {coordinates}")

    # url of obj
    obj_url = data['data']['welding_obj_file']

    welding_data = {'coordinates': coordinates, 'obj_url': obj_url}
    print("Coordinates are: ", welding_data['coordinates'])
    print("obj url is: ", welding_data["obj_url"])

    # Serialize welding data to JSON
    welding_data_json = json.dumps(welding_data)

    # Run simulation.py with the welding data as an argument using subprocess
    try:
        print(f"Starting simulation with data: {welding_data_json}")
        subprocess.run(['python', 'simulation.py', welding_data_json], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running simulation: {e}")
    # Extract the co-ordinates and run the code here : 

if __name__ == '__main__':
    sio.connect(PNC_SERVER)
    
    # Run the Flask app on localhost:5001
    # socketio.run(app, debug=False, use_reloader=False, port=PORT, host=HOST)
    # socketio.run(app, debug=False, port=5002, host="0.0.0.0")\
    try:
        sio.wait()
    except KeyboardInterrupt:
        print("\nExiting...")
        sio.disconnect()