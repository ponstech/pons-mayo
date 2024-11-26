# external package import
import cv2
import os
from img_enh import image_enhancement
# script_directory = os.path.dirname(os.path.abspath(__file__))
# os.chdir(script_directory)
def lambda_handler(event, context):
    # try:
    print(os.getcwd())
    # Decode the image array using cv2
    image = cv2.imread('output/example3_0.95.png')
    threshold = 0.7
    # print(image)
    #phase1, phase2, phase3, final = image_enhancement(image, threshold)

    new_image = image_enhancement(image, threshold)
    phase1 = new_image[:, :, 0]
    phase2 = new_image[:, :, 1]
    phase3 = new_image[:, :, 2]

    cv2.imwrite('output/test1.png', phase3)
    cv2.imwrite('output/test2.png', phase1)
    cv2.imwrite('output/test3.png', phase2)
    cv2.imwrite('output/test4.png', new_image)

    # except Exception as e:
    #     print(f"Error: {str(e)}")
    # #     return {
    # #         'statusCode': 500,
    # #         'body': str(e)
    # #     }


if __name__ == "__main__":
    lambda_handler({}, {})


# image_enhancement()
