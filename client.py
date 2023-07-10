import os
import socket
import concurrent.futures
import queue
import fire
import pyautogui
import time
import struct
import cv2
import numpy as np


def main(server_ip, server_port, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    def convert_jpeg_to_image(jpeg_data):
        # Convert the bytes to a numpy array
        nparr = np.frombuffer(jpeg_data, np.uint8)

        # Convert the numpy array to an image
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def receive_and_display_images(s, q):  # Add a queue parameter
        while True:
            data = b''
            payload_size = struct.calcsize('L')
            while len(data) < payload_size:
                data += s.recv(4096)
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack('L', packed_msg_size)[0]
            # Retrieve all data based on message size
            while len(data) < msg_size:
                data += s.recv(4096)
            frame_data = data[:msg_size]
            image = convert_jpeg_to_image(frame_data)
            q.put(image)  # Put the image on the queue instead of displaying it

    # Create a ThreadPoolExecutor
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    # Create a Queue
    q = queue.Queue()

    width, height = pyautogui.size()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print('Connecting...')
        s.connect((server_ip, server_port))
        print('Connected')

        # Start the receive_and_display_images function in another thread
        executor.submit(receive_and_display_images, s, q)  # Pass the queue to the function

        while True:
            x, y = pyautogui.position()  # Get the mouse position
            x = int(x / width * 19)
            y = int(y / height * 19)
            data = f'{x},{y}\n'
            s.sendall(data.encode('utf-8'))

            # Check if there's an image on the queue, and if so, display it
            if not q.empty():
                print('Displaying image')
                image = q.get()
                # Save the image to a file
                #date = time.strftime("%Y%m%d-%H%M%S")
                #filename = f"{output_folder}/{date}.png"
                #cv2.imwrite(filename, image)
                image = cv2.resize(image, (3000, 3000))
                cv2.imshow('window', image)
                cv2.waitKey(10)

            time.sleep(1)

if __name__ == "__main__":
    fire.Fire(main)
