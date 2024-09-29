import cv2
import numpy as np

points = []
pic_id = 2

class Projection(object):

    def __init__(self, image_path, points):
        """
            :param points: Selected pixels on top view(BEV) image
        """
        self.points = points

        if type(image_path) != str:
            self.image = image_path
        else:
            self.image = cv2.imread(image_path)
        self.height, self.width, self.channels = self.image.shape
    
    def cal_intrinsic_matrix(self, hov, h, w) -> np.ndarray:
        """
        Calculate the intrinsic matrix by given horizontal Of View in degrees and resolution 
        """

        vertical_fov = (h / w * hov) * np.pi / 180
        hov *= np.pi / 180

        f_x = (w / 2.0) / np.tan(hov / 2.0)
        f_y = (h / 2.0) / np.tan(vertical_fov / 2.0)

        K = np.array([[f_x, 0.0, w / 2.0], [0.0, f_y, h / 2.0], [0.0, 0.0, 1.0]])
        return K

    def axis_rotation_matrix(self, axis: str, angle: np.ndarray) -> np.ndarray:
        """
        Return the rotation matrix by given axis and eular angles.

        Args:
            axis: Axis label "X", "Y", or "Z".
            angle: numpy array of Euler angles in radians

        Returns:
            Rotation matrix (3 * 3).
        """

        cos = np.cos(angle)
        sin = np.sin(angle)
        one = np.ones_like(angle)
        zero = np.zeros_like(angle)

        if axis == "X":
            R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
        elif axis == "Y":
            R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
        elif axis == "Z":
            R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)

        return np.stack(R_flat, -1).reshape(angle.shape + (3, 3))


    def euler2matrix(self, euler_angles: np.ndarray, convention: str) -> np.ndarray:
        """
        Convert rotations given as Euler angles in radians to rotation matrix.

        Args:
            euler_angles: Euler angles (in radians)
            convention: Convention string of three uppercase letters from
                {"X", "Y", and "Z"}.

        Returns:
            Rotation matrix (3 * 3).
        """

        if euler_angles.ndim == 0 or euler_angles.shape[-1] != 3:
            raise ValueError("Invalid euler angles.")
        for letter in convention:
            if letter not in ("X", "Y", "Z"): 
                raise ValueError(f"Only XYZ")
        
        matrices = [
            self.axis_rotation_matrix(c, e)
            for c, e in zip(convention, np.moveaxis(euler_angles, -1, 0))
        ]
        
        return np.matmul(np.matmul(matrices[0], matrices[1]), matrices[2])

    def cal_transformation_matrix(self, p:list, euler_angle:list):
        '''
        Calculate the transformation matrix by given camera position and rotation (in radians)

        Return: 
            transformation matrix (4*4)
        '''
        H = np.eye(4)
        euler_angle = np.array(euler_angle)
        p = np.array(p)

        H[:3, :3] = self.euler2matrix(euler_angle, "XYZ")
        H[:3, -1] = p 

        return H

    def top_to_front(self, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0, fov=90):
        """
            Project the top view pixels to the front view pixels.
            :return: New pixels on perspective(front) view image
        """
        ### TODO ###
        cam1_H = self.cal_transformation_matrix(p = [0, 1, 0], euler_angle = [0, 0, 0])
        cam2_H = self.cal_transformation_matrix(p = [0, 2.5, 0], euler_angle = [-np.pi/2, 0, 0])

        cam1_intrin = self.cal_intrinsic_matrix(fov, 512, 512)
        cam2_intrin = self.cal_intrinsic_matrix(fov, 512, 512)

        related_H = np.linalg.inv(cam1_H) @ cam2_H # R_1 * R_related = R_2 => R_related = inverse(R_1) * R_2

        bev_points = np.array(self.points)
        bev_points_h = np.hstack((bev_points, np.ones((len(bev_points), 1))))
        bev_points_in_cam = np.linalg.inv(cam2_intrin) @ (bev_points_h.T * -cam2_H[1, -1]) # depth is negative to camera
        bev_points_in_cam = np.vstack((bev_points_in_cam, np.ones((1, len(bev_points)))))

        front_points_in_cam = related_H[:3, :4] @ bev_points_in_cam 
        front_points_in_pixel = cam1_intrin @ front_points_in_cam 
        front_points_in_pixel = np.round(front_points_in_pixel[:2] / front_points_in_pixel[2]).astype(int) 

        new_pixels = [[front_points_in_pixel[0, i], front_points_in_pixel[1, i]] for i in range(front_points_in_pixel.shape[1])]
        print("new_pixels: ", new_pixels)

        return new_pixels

    def show_image(self, new_pixels, img_name='projection.png', color=(0, 0, 255), alpha=0.4):
        """
            Show the projection result and fill the selected area on perspective(front) view image.
        """

        new_image = cv2.fillPoly(
            self.image.copy(), [np.array(new_pixels)], color)
        new_image = cv2.addWeighted(
            new_image, alpha, self.image, (1 - alpha), 0)

        cv2.imshow(
            f'Top to front view projection {img_name}', new_image)
        cv2.imwrite(img_name, new_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return new_image


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        print(x, ' ', y)
        points.append([x, y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, str(x) + ',' + str(y), (x+5, y+5), font, 0.5, (0, 0, 255), 1)
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('image', img)
        cv2.imwrite(filename=f"clicked{pic_id}.png", img=img)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:

        print(x, ' ', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        # cv2.putText(img, str(b) + ',' + str(g) + ',' + str(r), (x, y), font, 1, (255, 255, 0), 2)
        cv2.imshow('image', img)

if __name__ == "__main__":

    pitch_ang = -90

    front_rgb = f"bev_data/front{pic_id}.png"
    top_rgb = f"bev_data/bev{pic_id}.png"

    # click the pixels on window
    img = cv2.imread(top_rgb, 1)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    projection = Projection(front_rgb, points)
    new_pixels = projection.top_to_front(theta=pitch_ang)
    projection.show_image(new_pixels=new_pixels, img_name=f"projection{pic_id}.png")

