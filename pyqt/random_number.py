import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
import random

class RandomNumberGenerator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Random Number Generator")
        self.setGeometry(100, 100, 300, 200)

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFont(QFont("Arial", 20))
        self.label.setGeometry(50, 50, 200, 50)

        self.button = QPushButton("Generate", self)
        self.button.setGeometry(100, 120, 100, 50)
        self.button.clicked.connect(self.generate)

    def generate(self):
        random_number = random.randint(0, 99)
        self.label.setText(str(random_number))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = RandomNumberGenerator()
    window.show()
    sys.exit(app.exec_())
