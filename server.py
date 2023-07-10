import socket
import struct
import concurrent.futures
import fire
import cv2
import numpy as np
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch


def main(server_port, base_model_path, controlnet_path, control_image_path):

    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_path, controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    pipe.enable_xformers_memory_efficient_attention()
    control_image = load_image(control_image_path)
    generator = torch.manual_seed(0)


    def generate_image(prompt, steps=3):
        image = pipe(
            prompt, num_inference_steps=steps, generator=generator, image=control_image
        ).images[0]
        return image


    def convert_image_to_jpeg(image):
        return cv2.imencode('.jpg', image)[1].tobytes()


    # Create a ThreadPoolExecutor
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('0.0.0.0', server_port))
    s.listen()

    while True:
        conn, addr = s.accept()
        print(f"Connected by {addr}")
        while True:
            data = conn.recv(1024)
            if not data:
                break

            try:
                coords = data.decode().strip().split('\n')

                for coord in coords:
                    x, y = coord.split(',')

                    # Generate image in another thread and wait for it to complete
                    print(f"{x},{y}")
                    future = executor.submit(generate_image, f"{x},{y}", 3)
                    frame = future.result()

                    # Convert to cv2
                    frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

                    data = convert_image_to_jpeg(frame)

                    # Send message length first
                    message_size = struct.pack("L", len(data))

                    # Then data
                    conn.sendall(message_size + data)
            except Exception as e:
                print(e)
        conn.close()

if __name__ == "__main__":
    fire.Fire(main)
