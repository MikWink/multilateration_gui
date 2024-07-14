from PyQt5.QtWidgets import QApplication
from gui import MainWindow
import sys

def setup():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    print("Hallo")
    setup()

