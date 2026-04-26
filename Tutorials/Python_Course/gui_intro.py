import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout)
from PyQt5.QtGui import QIcon
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("My first cool GUI")
        self.setGeometry(700, 300, 500, 500)
        self.setWindowIcon(QIcon("08af45e32eb6c74832aee1daa9f790bd.jpg"))

        label = QLabel("Hello", self)
        label.setFont(QFont("Arial", 40))
        label.setGeometry(0, 0, 500, 100)
        label.setStyleSheet("color: blue;"
                            "background-color: red;"
                            "font-weight: bold;"
                            "font-style: italic;"
                            "text-decoration: underline;")
        label.setAlignment(Qt.AlignCenter)

        label = QLabel(self)
        label.setGeometry(0, 0, 250, 250)

        pixmap = QPixmap("08af45e32eb6c74832aee1daa9f790bd.jpg")
        label.setPixmap(pixmap)

        label.setScaledContents(True)
        label.setGeometry((self.width() - label.width()) // 2,
                          (self.height() - label.height()) // 2,
                          label.width(),
                          label.height())

        self.initUI()

    def initUI(self):
            central_widget = QWidget()
            self.setCentralWidget(central_widget)

            label1 = QLabel("#1")
            label2 = QLabel("#2")
            label3 = QLabel("#3")
            label4 = QLabel("#4")
            label5 = QLabel("#5")

            label1.setStyleSheet("background-color: red;")
            label2.setStyleSheet("background-color: yellow;")
            label3.setStyleSheet("background-color: blue;")
            label4.setStyleSheet("background-color: blue;")
            label5.setStyleSheet("background-color: purple;")

            vbox = QVBoxLayout()

            vbox.addWidget(label1)
            vbox.addWidget(label2)
            vbox.addWidget(label3)
            vbox.addWidget(label4)
            vbox.addWidget(label5)

            central_widget.setLayout((vbox))





def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()