import gymnasium as gym
import numpy as np
import cv2
import matplotlib.pyplot as plt

plt.style.use('ggplot')
def visualize_images(blur, canny):
    # Concatenate the images horizontally
    combined_image = np.hstack((blur, canny))

    # Show the combined image
    imS = cv2.resize(combined_image, (200, 300))
    cv2.imshow("Blur, Edges", imS)
    cv2.waitKey(1)


def find_error(observation, previous_error):
    def green_mask(observation):
        hsv = cv2.cvtColor(observation, cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(hsv, (36, 25, 25), (70, 255, 255))

        ## slice the green
        imask_green = mask_green > 0
        green = np.zeros_like(observation, np.uint8)
        green[imask_green] = observation[imask_green]
        return (green)

    def gray_scale(observation):
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        return gray

    def blur_image(observation):
        blur = cv2.GaussianBlur(observation, (5, 5), 0)
        return blur

    def canny_edge_detector(observation):
        canny = cv2.Canny(observation, 50, 150)
        return canny

    cropped = observation[63:65, 24:73]
    green = green_mask(cropped)
    grey = gray_scale(green)
    blur = blur_image(grey)
    canny = canny_edge_detector(blur)
    visualize_images(blur, canny)

    # find all non zero values in the cropped strip.
    # These non zero points(white pixels) corresponds to the edges of the road
    nz = cv2.findNonZero(canny)
    # horizontal cordinates of center of the road in the cropped slice
    mid = 24

    # some further adjustments obtained through trail and error
    if nz[:, 0, 0].max() == nz[:, 0, 0].min():
        if 30 > nz[:, 0, 0].max() > 20:
            return previous_error
        if nz[:, 0, 0].max() >= mid:
            return -15
        else:
            return +15
    else:
        return ((nz[:, 0, 0].max() + nz[:, 0, 0].min()) / 2) - mid


def pid(error, previous_error):
    Kp = 0.02
    Ki = 0.03
    Kd = 0.2

    steering = Kp * error + Ki * (error + previous_error) + Kd * (error - previous_error)
    # print("Steering", steering)
    return steering


env = gym.make('CarRacing-v2', render_mode="human")

observation = env.reset()
env.render()
rewardsum = 0
previous_error = 0

reward_arr = []

for x in [1, 0] * 500:
    try:
        error = find_error(observation, previous_error)

    except:
        error = -15
        # print("Error")
        pass

    steering = pid(error, previous_error)

    action = (steering, x, 0)

    observation, reward, truncated, terminated, info = env.step(action)
    previous_error = error
    rewardsum = rewardsum + reward
    print(observation)
    print("Reward", rewardsum)
    reward_arr.append(rewardsum)

    if terminated or truncated:
        env.close()
        cv2.destroyAllWindows()
        break

print(reward_arr)
plt.plot(reward_arr)
plt.title("Rewards v.s. episodes")
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.grid(True)
plt.show()
