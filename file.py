import cv2
import sys
import tkinter
from tkinter import filedialog
import numpy as np

class PlantDiseaseDetector:
    def __init__(self, master):
        self.master = master
        master.title("Plant Disease Detector")

        self.S = tkinter.Scale(master, from_=0, to=255, length=500, orient=tkinter.HORIZONTAL, background='white', fg='black', troughcolor='white', label="Processing Factor")
        self.S.pack()
        self.S.set(150)

        self.DiseasePercent = tkinter.StringVar()
        self.L = tkinter.Label(master, textvariable=self.DiseasePercent)
        self.L.pack()

        self.filename = self.get_file()
        if self.filename:
            self.process_image()
        else:
            print("No File!")
            exit(0)

    def get_file(self):
        if len(sys.argv) > 1:
            return sys.argv[1]
        else:
            return filedialog.askopenfilename(title="Select Image")

    def process_image(self):
        OriginalImage = cv2.imread(self.filename, 1)
        cv2.imshow("Original Image", OriginalImage)

        # Split channels
        b, g, r = cv2.split(OriginalImage)
        cv2.imshow("Red Channel", r)
        cv2.imshow("Green Channel", g)
        cv2.imshow("Blue Channel", b)

        # Alpha channel
        self.Alpha = self.get_alpha(OriginalImage)
        cv2.imshow("Alpha Channel", self.Alpha)

        # Disease detection
        Disease = r - g
        ProcessingFactor = self.S.get()
        for i in range(OriginalImage.shape[0]):
            for j in range(OriginalImage.shape[1]):
                if int(g[i, j]) > ProcessingFactor:
                    Disease[i, j] = 255
        cv2.imshow("Disease Image", Disease)
        self.display_disease_percentage(Disease)

        # Binding for reprocessing on Scale change
        self.S.bind('<ButtonRelease-1>', lambda event: self.process_image())

        # Additional transformations
        self.show_transformations(OriginalImage)

        self.master.mainloop()

    def get_alpha(self, OriginalImage):
        Alpha = np.zeros((OriginalImage.shape[0], OriginalImage.shape[1]), dtype=np.uint8)
        for i in range(OriginalImage.shape[0]):
            for j in range(OriginalImage.shape[1]):
                if all(OriginalImage[i, j] > 200):
                    Alpha[i, j] = 255
        return Alpha

    def display_disease_percentage(self, Disease):
        Count = 0
        Res = 0
        for i in range(Disease.shape[0]):
            for j in range(Disease.shape[1]):
                if self.Alpha[i, j] == 0:
                    Res += 1
                if Disease[i, j] < self.S.get():
                    Count += 1
        if Res > 0:
            Percent = (Count / Res) * 100
        else:
            Percent = 0
        self.DiseasePercent.set("Percentage Disease: " + str(round(Percent, 2)) + "%")

    def show_transformations(self, OriginalImage):
        # Edge detection using Canny
        edges = cv2.Canny(cv2.cvtColor(OriginalImage, cv2.COLOR_BGR2GRAY), 100, 200)
        cv2.imshow("Edges", edges)

        # Image blurring
        blurred = cv2.GaussianBlur(OriginalImage, (11, 11), 0)
        cv2.imshow("Blurred Image", blurred)

        # Scaling (downsampling)
        scaled = cv2.resize(OriginalImage, None, fx=0.5, fy=0.5)
        cv2.imshow("Scaled Image", scaled)

        # Translation
        rows, cols = OriginalImage.shape[:2]
        M = np.float32([[1, 0, 100], [0, 1, 50]])
        translated = cv2.warpAffine(OriginalImage, M, (cols, rows))
        cv2.imshow("Translated Image", translated)

        # Rotation
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
        rotated = cv2.warpAffine(OriginalImage, M, (cols, rows))
        cv2.imshow("Rotated Image", rotated)

        # Shearing
        M = np.float32([[1, 0.5, 0], [0.5, 1, 0]])
        sheared = cv2.warpAffine(OriginalImage, M, (cols, rows))
        cv2.imshow("Sheared Image", sheared)

        # Reflection
        reflected = cv2.flip(OriginalImage, 1)
        cv2.imshow("Reflected Image", reflected)

def main():
    MainWindow = tkinter.Tk()
    app = PlantDiseaseDetector(MainWindow)
    MainWindow.mainloop()

if __name__ == "__main__":
    main()
