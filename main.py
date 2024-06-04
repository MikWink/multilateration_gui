from PyQt5.QtWidgets import QApplication
from gui import MainWindow2
import sys

def setup():
    app = QApplication(sys.argv)
    window = MainWindow2()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    print("Hallo")
    setup()

