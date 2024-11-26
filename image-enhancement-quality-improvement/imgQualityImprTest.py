# external package import
import numpy as np
import cv2
from img_quality_impr import image_quality_impr
# from show import show
import os
def lambda_handler(event, context):
    try:
        # Decode the image array using cv2
        # print(os.getcwd())
        image = cv2.imread('input/example3.png')
        # print(image)
        result_img = image_quality_impr(image, 0.95)
        cv2.imwrite('output/example3_0.95.png', result_img)


    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': str(e)
        }




if __name__ == "__main__":
    lambda_handler({}, {})
# image_enhancement()
