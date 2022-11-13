"""
IDENTITY RECOGNITION AND APPLICATION ACTIONS FOR FALL DETECTION IN THE ELDERLY
Member: DAO DUY NGU, LE VAN THIEN
Mentor: PhD. TRAN THI MINH HANH
Time: 12/11/2022
contact: ddngu0110@gmail.com, ngocthien3920@gmail.com
"""
import sys
import PyQt5
import cv2
import time
import os
from PyQt5 import QtGui
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QTabWidget, QVBoxLayout, QMessageBox, QTableWidgetItem, QLabel
from PyQt5.QtGui import QIcon, QPixmap, QFont
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
from datetime import datetime, timedelta
from UI.main import Ui_Form as Tab_1
from UI.add_data import Ui_Form as Tab_2
from UI.show_database import Ui_Form as Show_database
from UI.show_history import Ui_Form as Show_history
# from Database.interface_sql import get_all_employee, update_info, delete_employee
from human_action_and_identity import ActionAndIdentityRecognition
from yolov5_face.detect_face import draw_result
from database.interface_sql import *


use_camera = 1  # tab 1 is allow using camera
camera_id = 0
url = 0
thread_1_running = False
width = 1536
height = 864
model_action = ActionAndIdentityRecognition()


def norm_size(w, h):
    return int(w*width), int(h*height)


class ActionThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)  # send image signal
    change_information_signal = pyqtSignal(dict)   # send detected information

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.model = model_action

    def run(self):
        global use_camera, url
        self.cap = cv2.VideoCapture(url)
        h_norm, w_norm = 720, 1280
        skip = True
        while self._run_flag and use_camera == 1:
            start = time.time()
            ret, frame = self.cap.read()
            if not ret:
                break
            h, w, _ = frame.shape
            if h > 1270 or w > w_norm:
                rate_max = max(h_norm / h, w_norm / w)
                frame = cv2.resize(frame, (int(rate_max * w), int(rate_max * h)), interpolation=cv2.INTER_AREA)
                h, w, _ = frame.shape
            frame, info = self.model.processing(frame, skip)
            skip = not skip
            fps = int(1 / (time.time() - start))
            cv2.putText(frame, 'FPS:' + str(fps), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            self.change_pixmap_signal.emit(frame)
            self.change_information_signal.emit(info)
        self.cap.release()
        self.stop()

    def stop(self):
        self._run_flag = False
        self.cap.release()
        self._run_flag = True


class AddFaceThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.model = model_action

    def run(self):
        global use_camera
        self.cap = cv2.VideoCapture(0)
        w_norm, h_norm = 1280, 720
        while use_camera == 2:
            ret, frame = self.cap.read()
            if not ret:
                break
            h, w, _ = frame.shape
            if h > 1270 or w > w_norm:
                rate_max = max(h_norm / h, w_norm / w)
                frame = cv2.resize(frame, (int(rate_max * w), int(rate_max * h)), interpolation=cv2.INTER_AREA)
                h, w, _ = frame.shape
            self.change_pixmap_signal.emit(frame)
        self.cap.release()


class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'Action Management Software For The Elderly'
        self.setWindowIcon(QIcon('icon/logo.png'))
        self.left = 0
        self.top = 0
        self.width = width
        self.height = height
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.table_widget = MyTableWidget(self)
        self.table_widget.tabs.currentChanged.connect(self.on_click)
        self.setCentralWidget(self.table_widget)
        self.show()

    def on_click(self):
        global use_camera, thread_1_running
        if self.table_widget.tabs.currentIndex() == 0:
            thread_1_running = False
            use_camera = 1
        else:
            use_camera = 2
            self.table_widget.tab2.run()


class MyTableWidget(QWidget):

    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        self.layout = QVBoxLayout(self)

        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tab1 = Camera()
        self.tab2 = AddPeople()
        # self.tab3 = ShowDatabase()
        # Add tabs
        self.tabs.addTab(self.tab1, "Camera")
        self.tabs.addTab(self.tab2, "Add Face")
        # self.tabs.addTab(self.tab3, "Show database")
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)


class Camera(QWidget, Tab_1):
    def __init__(self, parent=None):
        super(Camera, self).__init__(parent)
        self.setupUi(self)
        self.screen_width, self.screen_height = 1366, 768
        self.label_screen.resize(self.screen_width, self.screen_height)
        self.width_sub_screen, self.height_sub_sceen = 200, 200
        self.image_action.setPixmap(QtGui.QPixmap("icon/unknown_person.jpg").scaled(200, 200))
        self.run_button.setIcon(QIcon('icon/play.png'))
        self.run_button.clicked.connect(self.run)
        self.stop_button.setIcon(QIcon('icon/pause.png'))
        self.stop_button.clicked.connect(self.stop)
        self.logo_dhbk.setPixmap(QtGui.QPixmap('icon/Logodhbk.jpg').scaled(200, 200))
        self.logo_dtvt.setPixmap(QtGui.QPixmap('icon/logo_dtvt.jpg').scaled(200, 200))
        self.rtsp_video.setText('0')
        self.thread = ActionThread()
        self.thread.change_pixmap_signal.connect(self.update_image_main_screen)
        self.thread.change_information_signal.connect(self.update_data)
        self.show()

    def run(self):
        global use_camera, thread_1_running, url
        if use_camera != 1:
            use_camera = 1
        thread_1_running = False
        if not thread_1_running:

            url = self.rtsp_video.text()
            if len(url) == 0:
                QMessageBox.warning(self, "url not find", "warning")
                self.stop()
            else:
                if url == '0':
                    url = 0  # turn on webcam
                self.thread.start()
            thread_1_running = True

    def stop(self):
        global thread_1_running, use_camera
        try:
            use_camera = 2
            self.thread.stop()
            thread_1_running = False
        except:
            pass

    @pyqtSlot(np.ndarray)
    def update_data(self, data):
        try:
            now = datetime.now()
            image = data['image']
            qt_img = self.convert_cv_qt(image, self.width_sub_screen, self.height_sub_sceen)
            if data['name'] == 'Unknown':
                self.image_action.setPixmap(QtGui.QPixmap('icon/unknown_person.jpg').scaled(200, 200))
            else:
                self.image_action.setPixmap(self.convert_cv_qt(data['image'], 200, 200))
            self.image_action.setPixmap(qt_img)
            self.name_people.setText(data['name'])
            self.name_action.setText(data['action'])
            self.time_action.setText(now.strftime('%a %H:%M:%S'))
            # save database
            add_action(data_tuple=(data['id'], data['name'], data['image'], data['action'], now.strftime('%a %H:%M:%S')),
                       name_table='action_data')
        except Exception:
            pass

    def update_image_main_screen(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img, self.screen_width, self.screen_height)
        self.label_screen.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img, w_screen, h_screen):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        rgb_image = cv2.resize(rgb_image, (w_screen, h_screen))
        # rgb_image = cv2.flip(rgb_image, flipCode=1)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(w_screen, h_screen, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


class AddPeople(QWidget, Tab_2):
    def __init__(self):
        super(AddPeople, self).__init__()
        self.setupUi(self)
        self.run_button.setIcon(QIcon('icon/play.png'))
        self.run_button.clicked.connect(self.recog)
        self.stop_button.setIcon(QIcon('icon/pause.png'))
        self.stop_button.clicked.connect(self.stop)
        self.save_data.setIcon(QIcon('icon/save.png'))
        self.save_data.clicked.connect(self.save)
        self.logo_dhbk.setPixmap(QtGui.QPixmap('icon/Logodhbk.jpg').scaled(200, 200))
        self.logo_dtvt.setPixmap(QtGui.QPixmap('icon/logo_dtvt.jpg').scaled(200, 200))
        self.button_show_database.clicked.connect(self.show_tab_database)
        self.button_show_history.clicked.connect(self.show_tab_history)
        self.press(None)
        self.face_model = model_action.face_model
        self.recog_flag = False
        self.list_image = []
        self.thread = AddFaceThread()
        self.thread.change_pixmap_signal.connect(self.update_image_main_screen)
        self.thread_cap_run = False
        # self.run()
        self.show()

    def run(self):
        global use_camera
        self.thread_cap_run = False
        if use_camera != 2:
            use_camera = 2
        time.sleep(0.5)
        if self.thread_cap_run is False:
            self.thread.start()
            self.thread_cap_run = True

    def press(self, event):
        self.id.clear()
        self.name.clear()

    def show_tab_database(self):
        self.show_tab = ShowDatabase()

    def show_tab_history(self):
        self.show_tab1 = ShowHistory()

    def recog(self):
        if self.name.text().strip() != '':
            self.recog_flag = True
        else:
            QMessageBox.warning(self, 'Warning!', 'Fill name first')

    def stop(self):
        self.recog_flag = False

    def save(self):
        try:
            ret = QMessageBox.question(self, 'confirm', "Do you want to save?\n(Yes) or (No)",
                                       QMessageBox.No | QMessageBox.Yes)
            if ret == QMessageBox.Yes:
                self.recog_flag = False
                name = self.name.text()
                id = self.id.text()
                self.id.setText('')
                self.name.setText('')
                self.face_model.create_data(self.list_image, name, id)
                QMessageBox.warning(self, 'Completed!', '')
                self.list_image = []
            else:
                pass
        except:
            QMessageBox.warning(self, 'Warning!', "Can't create data")
            self.list_image = []

    @pyqtSlot(np.ndarray)
    def update_image_main_screen(self, cv_img):
        """Updates the image_label with a new opencv image"""
        self.image = cv_img.copy()
        bbox, label, label_id, score, landmark = self.face_model.detect(cv_img)
        if len(bbox) != 0:
            if self.recog_flag:
                self.list_image.append(self.image.copy())
        for idx, box in enumerate(bbox):
            draw_result(cv_img, box, '', score[idx], landmark[idx])
        qt_img = self.convert_cv_qt(cv_img, 1366, 768)
        self.label_screen.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img, w_screen, h_screen):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        rgb_image = cv2.resize(rgb_image, (w_screen, h_screen))
        rgb_image = cv2.flip(rgb_image, flipCode=1)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(w_screen, h_screen, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


class ShowDatabase(QWidget, Show_database):
    def __init__(self):
        super(ShowDatabase, self).__init__()
        self.setupUi(self)
        self.setWindowTitle("Database")
        self.delete_database.clicked.connect(self.delete_current)
        self.save_database.clicked.connect(self.save_change)
        self.table_database.horizontalHeader().setDefaultSectionSize(224)
        self.table_database.verticalHeader().setDefaultSectionSize(224)
        font = QFont()
        font.setPointSize(15)
        self.table_database.setFont(font)
        self.show_database()
        self.show()

    def show_database(self):
        data_face = get_all_face('faceid')
        self.table_database.setRowCount(len(data_face[0]))
        for index, data in enumerate(data_face[:len(data_face)-1]):
            for idx, header in enumerate(data):
                if index == (len(data_face) - 2):
                    image = QLabel("")
                    image.setScaledContents(True)
                    image.setPixmap(self.convert_cv_qt(header, 224, 224))
                    self.table_database.setCellWidget(idx, index, image)
                else:
                    self.table_database.setItem(idx, index, QTableWidgetItem(str(header)))

    def save_change(self):
        ret = QMessageBox.question(self, 'Warning',
                                   "Are you sure you want to change? \n Ok (Yes) or No (No)",
                                   QMessageBox.No | QMessageBox.Yes)

        if ret == QMessageBox.Yes:
            for row in range(self.table_database.rowCount()):
                fix = []
                for col in range(self.table_database.columnCount()):
                    if col == 2:
                        fix.append(int(self.table_database.item(row, col).text()))
                        continue
                    fix.append(self.table_database.item(row, col).text())
                update_info(tuple(fix))
            QMessageBox.warning(self, "Save successfully", "Completed.")
        self.show_database()

    def convert_cv_qt(self, cv_img, w_screen, h_screen):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        rgb_image = cv2.resize(rgb_image, (w_screen, h_screen))
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(w_screen, h_screen, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def delete_current(self):
        ret = QMessageBox.question(self, 'Warning',
                                   "Are you sure you want to delete? \n Ok (Yes) or No (No)",
                                   QMessageBox.No | QMessageBox.Yes)
        if ret == QMessageBox.Yes:
            x = self.table_database.currentRow()
            code_id = self.table_database.item(x, 0).text()
            delete_face(code_id)
            self.show_database()
            QMessageBox.warning(self, "Delete successfully", "Completed.")


class ShowHistory(QWidget, Show_history):
    def __init__(self):
        super(ShowHistory, self).__init__()
        self.setupUi(self)
        self.setWindowTitle("Database")
        self.table_database.horizontalHeader().setDefaultSectionSize(224)
        self.table_database.verticalHeader().setDefaultSectionSize(224)
        font = QFont()
        font.setPointSize(15)
        self.table_database.setFont(font)
        self.show_history()
        self.show()

    def show_history(self):
        data_face = get_all_action('action_data')
        self.table_database.setRowCount(len(data_face[0]))
        for index, data in enumerate(data_face):
            for idx, header in enumerate(data):
                if index == 2:
                    image = QLabel("")
                    image.setScaledContents(True)
                    image.setPixmap(self.convert_cv_qt(header, 224, 224))
                    self.table_database.setCellWidget(idx, index, image)
                else:
                    self.table_database.setItem(idx, index, QTableWidgetItem(str(header)))

    def convert_cv_qt(self, cv_img, w_screen, h_screen):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        rgb_image = cv2.resize(rgb_image, (w_screen, h_screen))
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(w_screen, h_screen, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())