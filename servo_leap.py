import leap
import numpy as np
import cv2
import time
import paho.mqtt.client as mqtt

# Initialize timestamp
timestamp = time.time()

# MQTT setup
BROKER_ADDRESS = "leapservo.ddns.net"
MQTT_TOPIC = "leap/servo"

# Set up the MQTT client
client = mqtt.Client()
client.connect(BROKER_ADDRESS, 1883, 60)

# Dictionary to map Leap Motion tracking modes to strings
_TRACKING_MODES = {
    leap.TrackingMode.Desktop: "Desktop",
    leap.TrackingMode.HMD: "HMD",
    leap.TrackingMode.ScreenTop: "ScreenTop",
}

class Canvas:
    def __init__(self):
        # Initialize the Canvas object with default parameters
        self.name = "Python Gemini Visualiser"
        self.screen_size = [500, 700]
        self.hands_colour = (144, 238, 144)  # Color for hand visualization
        self.font_colour = (0, 0, 255)       # Color for text
        self.hands_format = "Skeleton"       # Initial hand visualization format
        self.output_image = np.zeros((self.screen_size[0], self.screen_size[1], 3), np.uint8)
        self.tracking_mode = None

    def set_tracking_mode(self, tracking_mode):
        # Set the tracking mode
        self.tracking_mode = tracking_mode

    def toggle_hands_format(self):
        # Toggle hand visualization format between Skeleton and Dots
        self.hands_format = "Dots" if self.hands_format == "Skeleton" else "Skeleton"
        print(f"Set hands format to {self.hands_format}")
    
    def get_joint_position(self, bone):
        # Get the position of a joint in the output image
        if bone:
            return int(bone.x + (self.screen_size[1] / 2)), int(bone.z + (self.screen_size[0] / 2))
        else:
            return None

    def render_hands(self, event):
        # Clear the previous image
        self.output_image[:, :] = 0

        # Display the current tracking mode on the image
        cv2.putText(
            self.output_image,
            f"Tracking Mode: {_TRACKING_MODES[self.tracking_mode]}",
            (10, self.screen_size[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.font_colour,
            1,
        )

        # If no hands are detected, return early
        if len(event.hands) == 0:
            return

        for i in range(len(event.hands)):
            hand = event.hands[i]

            global timestamp
            currentTmstmp = time.time()

            # Process the first detected hand and send position data via MQTT every 0.2 seconds
            if i == 0 and (currentTmstmp - timestamp) > 0.2:
                x0 = hand.palm.position.x
                y0 = hand.palm.position.y
                z0 = hand.palm.position.z

                # Define bounds for x and z positions
                X_MAX = 180
                X_MIN = -180
                Z_MAX = 200
                Z_MIN = -100

                # Constrain x0 within bounds
                if x0 < X_MIN:
                    x0 = X_MIN
                elif x0 > X_MAX:
                    x0 = X_MAX
                
                # Constrain z0 within bounds
                if z0 < Z_MIN:
                    z0 = Z_MIN
                elif z0 > Z_MAX:
                    z0 = Z_MAX

                # Normalize x0 range
                x0 = x0 + (-1) * X_MIN
                x0 = x0 / 2

                # Normalize z0 range
                z0 = z0 + (-1) * Z_MIN
                zk = z0 / 300
                z0 = int(zk * 180)

                print(f"x: {x0:.2f}, y: {y0:.2f}, z: {z0:.2f}")

                servo_x = int(x0)
                servo_z = int(z0)

                # Publish the position data to the MQTT broker
                msg = f"{servo_x},{servo_z}\n"
                client.publish(MQTT_TOPIC, msg)
                timestamp = currentTmstmp

            for index_digit in range(5):
                digit = hand.digits[index_digit]
                for index_bone in range(4):
                    bone = digit.bones[index_bone]
                    
                    if self.hands_format == "Dots":
                        # Draw dots for the joints
                        prev_joint = self.get_joint_position(bone.prev_joint)
                        next_joint = self.get_joint_position(bone.next_joint)
                        if prev_joint:
                            cv2.circle(self.output_image, prev_joint, 2, self.hands_colour, -1)
                        if next_joint:
                            cv2.circle(self.output_image, next_joint, 2, self.hands_colour, -1)
                    
                    if self.hands_format == "Skeleton":
                        # Draw lines for the skeleton
                        wrist = self.get_joint_position(hand.arm.next_joint)
                        elbow = self.get_joint_position(hand.arm.prev_joint)
                        if wrist:
                            cv2.circle(self.output_image, wrist, 3, self.hands_colour, -1)
                        if elbow:
                            cv2.circle(self.output_image, elbow, 3, self.hands_colour, -1)
                        if wrist and elbow:
                            cv2.line(self.output_image, wrist, elbow, self.hands_colour, 2)
                        bone_start = self.get_joint_position(bone.prev_joint)
                        bone_end = self.get_joint_position(bone.next_joint)
                        if bone_start:
                            cv2.circle(self.output_image, bone_start, 3, self.hands_colour, -1)
                        if bone_end:
                            cv2.circle(self.output_image, bone_end, 3, self.hands_colour, -1)
                        if bone_start and bone_end:
                            cv2.line(self.output_image, bone_start, bone_end, self.hands_colour, 2)
                        # Connect adjacent bones for skeleton visualization
                        if ((index_digit == 0) and (index_bone == 0)) or (
                            (index_digit > 0) and (index_digit < 4) and (index_bone < 2)
                        ):
                            index_digit_next = index_digit + 1
                            digit_next = hand.digits[index_digit_next]
                            bone_next = digit_next.bones[index_bone]
                            bone_next_start = self.get_joint_position(bone_next.prev_joint)
                            if bone_start and bone_next_start:
                                cv2.line(
                                    self.output_image,
                                    bone_start,
                                    bone_next_start,
                                    self.hands_colour,
                                    2,
                                )
                        if index_bone == 0 and bone_start and wrist:
                            cv2.line(self.output_image, bone_start, wrist, self.hands_colour, 2)

class TrackingListener(leap.Listener):
    def __init__(self, canvas):
        self.canvas = canvas

    def on_connection_event(self, event):
        # Callback for connection event
        print("Connected to Leap Motion")

    def on_tracking_mode_event(self, event):
        # Callback for tracking mode change event
        self.canvas.set_tracking_mode(event.current_tracking_mode)
        print(f"Tracking mode changed to {_TRACKING_MODES[event.current_tracking_mode]}")

    def on_device_event(self, event):
        # Callback for device event
        try:
            with event.device.open():
                info = event.device.get_info()
        except leap.LeapCannotOpenDeviceError:
            info = event.device.get_info()

        print(f"Found device {info.serial}")

    def on_tracking_event(self, event):
        # Callback for tracking event
        self.canvas.render_hands(event)

def main():
    # Initialize the Canvas and TrackingListener
    canvas = Canvas()

    print(canvas.name)
    print("")
    print("Press <key> in visualiser window to:")
    print("  x: Exit")
    print("  h: Select HMD tracking mode")
    print("  s: Select ScreenTop tracking mode")
    print("  d: Select Desktop tracking mode")
    print("  f: Toggle hands format between Skeleton/Dots")

    tracking_listener = TrackingListener(canvas)

    # Establish a connection with the Leap Motion Controller
    connection = leap.Connection()
    connection.add_listener(tracking_listener)

    running = True

    with connection.open():
        connection.set_tracking_mode(leap.TrackingMode.Desktop)
        canvas.set_tracking_mode(leap.TrackingMode.Desktop)

        while running:
            # Display the output image in a window
            cv2.imshow(canvas.name, canvas.output_image)

            key = cv2.waitKey(1)

            if key == ord("x"):
                break
            elif key == ord("h"):
                connection.set_tracking_mode(leap.TrackingMode.HMD)
            elif key == ord("s"):
                connection.set_tracking_mode(leap.TrackingMode.ScreenTop)
            elif key == ord("d"):
                connection.set_tracking_mode(leap.TrackingMode.Desktop)
            elif key == ord("f"):
                canvas.toggle_hands_format()
        
    # Close the MQTT client connection
    client.disconnect()

if __name__ == "__main__":
    main()
