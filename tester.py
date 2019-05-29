from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QDesktopWidget, QGridLayout, QLabel, QPushButton
from PyQt5.QtGui import QPainter, QColor, QBrush, QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, QObject, pyqtSignal, pyqtSlot
from os import listdir
from os.path import isfile, join
from numpy import random
from PIL import Image
import time

PIC_SIDE_CM = 15.24
CM_PER_INCH = 2.54
PIXELS_PER_INCH = 109
IMAGES_PATH = './tester_images'


class Window(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Adversarial examples tester')
        self.setFixedSize(360, 120)
        self.setCentralWidget(ControlView())


class ControlView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        images = [f for f in listdir(IMAGES_PATH) if isfile(
            join(IMAGES_PATH, f)) and f.endswith('.jpg')]
        self.images = list(map(lambda img: join(IMAGES_PATH, img), images))
        grid = QGridLayout(self)
        images_count_label = QLabel('{} images found'.format(len(images)))
        grid.addWidget(images_count_label, 0, 0)
        start_button = QPushButton('Start test')
        start_button.clicked.connect(self.on_start_button_clicked)
        grid.addWidget(start_button, 0, 1)
        self.setLayout(grid)

    def on_start_button_clicked(self):
        test_window = TestWindow(self.images, self)
        test_window.showFullScreen()


class TestWindow(QMainWindow):
    def __init__(self, images, parent=None):
        super().__init__(parent)
        self.set_bg_color()
        self.setCursor(Qt.BlankCursor)
        self.current_image = None
        self.already_chosen = False
        self.choices = []
        self.thread = QThread()
        self.slide_show_worker = SlideShowWorker(images)
        self.slide_show_worker.moveToThread(self.thread)
        self.thread.started.connect(self.slide_show_worker.run)
        self.slide_show_worker.end.connect(self.thread.quit)
        self.slide_show_worker.fixation.connect(self.show_fixation)
        self.slide_show_worker.image.connect(self.show_image)
        self.slide_show_worker.mask.connect(self.show_mask)
        self.slide_show_worker.dark_screen.connect(self.show_dark_screen)
        self.slide_show_worker.end.connect(self.end_test)
        self.thread.start()

    def show_fixation(self):
        self.current_image = None
        self.already_chosen = False
        self.setCentralWidget(FixationView(self))

    def show_image(self, image):
        self.current_image = image
        self.setCentralWidget(ImageView(self, image))

    def show_mask(self):
        self.setCentralWidget(MaskView(self))

    def show_dark_screen(self):
        self.setCentralWidget(None)

    def end_test(self):
        print('Choices:')
        for choice in self.choices:
            print(choice)
        self.close()

    def set_bg_color(self):
        palette = self.palette()
        palette.setColor(self.backgroundRole(), Qt.black)
        self.setPalette(palette)

    def keyPressEvent(self, event):
        if self.current_image is not None:
            if self.already_chosen is False:
                if event.key() == Qt.Key_Left:
                    self.already_chosen = True
                    self.choices.append((self.current_image, 1))
                    print('class 1 chosen')
                elif event.key() == Qt.Key_Right:
                    self.already_chosen = True
                    self.choices.append((self.current_image, 2))
                    print('class 2 chosen')
                else:
                    print('Wrong button!')
            else:
                print('Already chosen class for that image!')
        else:
            print('Pressed in wrong moment!')


class SlideShowWorker(QObject):

    fixation = pyqtSignal()
    image = pyqtSignal(str)
    mask = pyqtSignal()
    dark_screen = pyqtSignal()
    end = pyqtSignal()

    def __init__(self, image_paths):
        QObject.__init__(self)
        self.image_paths = image_paths

    @pyqtSlot()
    def run(self):
        time.sleep(5)
        for img in self.image_paths:
            self.fixation.emit()
            time.sleep(1)
            self.image.emit(img)
            time.sleep(0.063)
            self.mask.emit()
            time.sleep(0.02)
            self.dark_screen.emit()
            time.sleep(2.5)
        self.end.emit()


class FixationView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

    def paintEvent(self, e):
        painter = QPainter()
        painter.begin(self)
        painter.setPen(Qt.white)
        painter.setBrush(Qt.white)
        window_w = self.frameGeometry().width()
        window_h = self.frameGeometry().height()
        rect_w = 6
        rect_h = 40
        painter.drawRect(window_w/2 - rect_w/2, window_h /
                         2 - rect_h/2, rect_w, rect_h)
        painter.drawRect(window_w/2 - rect_h/2, window_h /
                         2 - rect_w/2, rect_h, rect_w)
        painter.end()


class ImageView(QWidget):
    def __init__(self, parent=None, image_path=None):
        super().__init__(parent)
        self.picture = QLabel(self)
        if image_path is not None:
            self.set_image(image_path)

    def set_image(self, image_path):
        pic_side = int((PIC_SIDE_CM / CM_PER_INCH) * PIXELS_PER_INCH)
        self.picture.resize(pic_side, pic_side)
        self.picture.setPixmap(QPixmap(image_path).scaled(pic_side, pic_side))
        x = QDesktopWidget().screenGeometry(-1).width() / 2 - pic_side / 2
        y = QDesktopWidget().screenGeometry(-1).height() / 2 - pic_side / 2
        self.picture.move(x, y)
        self.picture.show()


class MaskView(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        res = int((PIC_SIDE_CM / CM_PER_INCH) * PIXELS_PER_INCH)
        mask = random.choice(a=[0, 255], size=(res, res))
        image = QImage(mask.astype('uint8'), res, res, QImage.Format_Indexed8)
        self.picture = QLabel(self)
        self.picture.setPixmap(QPixmap.fromImage(image))
        x = QDesktopWidget().screenGeometry(-1).width() / 2 - res/2
        y = QDesktopWidget().screenGeometry(-1).height() / 2 - res/2
        self.picture.move(x, y)
        self.picture.show()


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
