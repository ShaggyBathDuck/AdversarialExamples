from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QDesktopWidget, QGridLayout, QLabel, QPushButton
from PyQt5.QtGui import QPainter, QColor, QBrush, QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, QObject, pyqtSignal, pyqtSlot
from os import listdir
from os.path import isfile, join, isdir
from numpy import random
from PIL import Image
import time

PIC_SIDE_CM = 15.24
CM_PER_INCH = 2.54
PIXELS_PER_INCH = 109
IMAGES_PATH = './tester_images'
IMAGES_COUNTS = [(2, 1), (2, 1)]


class Image:
    def __init__(self, file, cls, original):
        self.file = file
        self.cls = cls
        self.original = original
        self.guessed_cls = None

    def is_guessed_correctly(self):
        return self.guessed_cls == self.cls


class Tester:
    def __init__(self, images_path, images_counts):
        self._images_path = images_path
        self._last_guessed_index = -1
        directories = [d for d in listdir(
            images_path) if isdir(join(images_path, d))]
        self.classes = [d for d in directories if not d.endswith('_adv')]
        if len(self.classes) != 2:
            raise RuntimeError(
                'There should be exactly 2 directories named after image classes, found: {0}'.format(self.classes))
        self._chosen_images = []
        for i, cls in enumerate(self.classes):
            self._collect_images_for_class(i, cls, images_counts[i])
        random.shuffle(self._chosen_images)
        self.images_count = len(self._chosen_images)

    def _collect_images_for_class(self, cls, label, images_count):
        original_dir = join(self._images_path, label)
        original_images = self._list_image_files(original_dir)
        self._print_image_count(len(original_images), label)

        adv_dir = join(self._images_path, label + '_adv')
        if not isdir(adv_dir):
            raise RuntimeError(
                'Missing adversarial images directory for class {0}'.format(label))
        adv_images = self._list_image_files(adv_dir)
        self._print_image_count(len(adv_images), label + ' adversarial')

        chosen_original_images = random.choice(
            a=original_images, size=images_count[0], replace=False)
        for img in chosen_original_images:
            self._chosen_images.append(Image(join(original_dir, img), cls, True))
        chosen_adv_images = random.choice(
            a=adv_images, size=images_count[1], replace=False)
        for img in chosen_adv_images:
            self._chosen_images.append(Image(join(adv_dir, img), cls, False))

    def _list_image_files(self, dir):
        return [f for f in listdir(dir) if isfile(join(dir, f)) and (f.endswith('.jpg') or f.endswith('.jpeg'))]

    def _print_image_count(self, count, label):
        print('{0} {1} images found'.format(count, label))

    def has_next_image(self):
        return self._last_guessed_index < len(self._chosen_images - 1)

    def next_image(self):
        self._last_guessed_index += 1
        return self._chosen_images[self._last_guessed_index].file

    def guess(self, cls):
        image = self._chosen_images[self._last_guessed_index]
        if image.guessed_cls is None:
            image.guessed_cls = cls

    def print_accurracy(self):
        general_acc = len([img for img in self._chosen_images if img.is_guessed_correctly(
        )]) / len(self._chosen_images)
        print('General accurracy is {}'.format(general_acc))
        for cls,label in enumerate(self.classes):
            original_class_images = [
                img for img in self._chosen_images if img.cls == cls and img.original]
            original_class_acc = len(
                [img for img in original_class_images if img.is_guessed_correctly()]) / len(original_class_images)
            print('{} accurracy is {}'.format(label, original_class_acc))

            adv_images = [
                img for img in self._chosen_images if img.cls == cls and not img.original]
            if len(adv_images) > 0:
                adv_acc = len(
                    [img for img in adv_images if img.is_guessed_correctly()]) / len(adv_images)
                print('Adversarial {} accurracy is {}'.format(label, adv_acc))


class Window(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Adversarial examples tester')
        self.setFixedSize(480, 120)
        self.setCentralWidget(ControlView())


class ControlView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.tester = Tester(IMAGES_PATH, IMAGES_COUNTS)        
        grid = QGridLayout(self)
        first_class_help_label = QLabel('Left arrow <- {}'.format(self.tester.classes[0]))
        second_class_help_label = QLabel('Right arrow -> {}'.format(self.tester.classes[1]))
        start_button = QPushButton('Start test')
        start_button.clicked.connect(self.on_start_button_clicked)
        grid.addWidget(first_class_help_label, 0, 0)
        grid.addWidget(second_class_help_label, 1, 0)
        grid.addWidget(start_button, 0, 1, 2, 1)
        self.setLayout(grid)

    def on_start_button_clicked(self):
        test_window = TestWindow(self.tester, self)
        test_window.showFullScreen()


class TestWindow(QMainWindow):
    def __init__(self, tester, parent=None):
        super().__init__(parent)
        self.tester = tester
        self.set_bg_color()
        self.setCursor(Qt.BlankCursor)
        self.current_image = None
        self.already_chosen = False
        self.choices = []
        self.thread = QThread()
        self.slide_show_worker = SlideShowWorker(self.tester.images_count)
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

    def show_image(self):
        self.current_image = self.tester.next_image()
        print('Showing {}'.format(self.current_image))
        self.setCentralWidget(ImageView(self, self.current_image))

    def show_mask(self):
        self.setCentralWidget(MaskView(self))

    def show_dark_screen(self):
        self.setCentralWidget(None)

    def end_test(self):
        self.tester.print_accurracy()
        print('Choices:')
        for choice in self.choices:
            print(choice)
        self.close()

    def set_bg_color(self):
        palette = self.palette()
        palette.setColor(self.backgroundRole(), Qt.black)
        self.setPalette(palette)

    def on_image_guessed(self, cls):
        self.already_chosen = True
        self.choices.append((self.current_image, cls))
        self.tester.guess(cls)

    def keyPressEvent(self, event):
        if self.current_image is not None:
            if self.already_chosen is False:
                if event.key() == Qt.Key_Left:
                    self.on_image_guessed(0)
                elif event.key() == Qt.Key_Right:
                    self.on_image_guessed(1)
                else:
                    print('Wrong button!')
            else:
                print('Already chosen class for that image!')
        else:
            print('Pressed in wrong moment!')


class SlideShowWorker(QObject):

    fixation = pyqtSignal()
    image = pyqtSignal()
    mask = pyqtSignal()
    dark_screen = pyqtSignal()
    end = pyqtSignal()

    def __init__(self, image_count):
        QObject.__init__(self)
        self.image_count = image_count

    @pyqtSlot()
    def run(self):
        time.sleep(5)
        for i in range(self.image_count):
            self.fixation.emit()
            time.sleep(1)
            self.image.emit()
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
