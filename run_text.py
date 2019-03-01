import sys
from PyQt5.QtWidgets import (QWidget, QToolTip, QPushButton, QApplication, QMessageBox, QDesktopWidget, QScrollArea,
                             QFileDialog, QHBoxLayout, QTableWidget, QTableWidgetItem,QMainWindow,
                             QLineEdit, QGridLayout, QLCDNumber, QSlider, QVBoxLayout, QInputDialog, QSizePolicy, QDialog)
from PyQt5.QtGui import QFont, QIcon, QColor, QPixmap, QScreen
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5 import QtWidgets, QtCore, QtGui
import pandas as pd
import numpy as np
from PandasModel import PandasModel
import pickle
from Predict import Predict


class TalkBackWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowIcon(QtGui.QIcon('Pirate.png'))
        # self.setAutoFillBackground(True)
        # p = self.palette()
        # p.setColor(self.backgroundRole(), Qt.white)
        # self.setPalette(p)
        load_btn = QPushButton('Load data', self)
        predict_btn = QPushButton('Predict', self)
        present_btn = QPushButton('Present', self)
        present_worst_btn = QPushButton('Present worst', self)

        quit_btn = QPushButton('Quit', self)
        vbox = QVBoxLayout()
        vbox.addWidget(load_btn)
        vbox.addWidget(predict_btn)
        vbox.addWidget(present_btn)
        vbox.addWidget(present_worst_btn)
        vbox.addWidget(quit_btn)
        self.setLayout(vbox)
        self.setGeometry(screen_width/3, screen_height/3, screen_width/3, screen_height/3)
        self.center()
        self.setWindowTitle('Talk back classifier')

        load_btn.clicked.connect(self.load_button_clicked)
        predict_btn.clicked.connect(self.predict_button_clicked)
        present_btn.clicked.connect(self.present_btn_clicked)
        present_worst_btn.clicked.connect(self.present_worst_btn_clicked)


        quit_btn.clicked.connect(QApplication.instance().quit)

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def predict_button_clicked(self):
        """ Model prediction here"""
        try:
            with open('nb_pickle_model.pkl', 'rb') as file:
                nb_model_obj = pickle.load(file)
            with open('lr_pickle_model.pkl', 'rb') as file:
                lr_model_obj = pickle.load(file)
            predict_obj = Predict(nb_model_obj, lr_model_obj)
            self.data = predict_obj.ui_predcit_nb(self.data)
            #self.data['Prediction'] = pd.Series(np.random.randn(len(self.data['ARTICLE_ID'])))
            print("Prediction completed")
        except Exception as e:
            print(e)
        return

    def present_btn_clicked(self):
        try:
            # sorted_data = self.data.sort_values(by=['Prediction'], ascending=False)
            self.worst = False
            if not hasattr(self, 'DataPresentor'):
                self.data_window = DataPresentor(self, False)
            self.data_window.show()
            self.hide()
        except Exception as e:
            print(e)

    def present_worst_btn_clicked(self):
        try:
            # sorted_data = self.data.sort_values(by=['Prediction'], ascending=True)
            self.worst = True
            if not hasattr(self, 'DataPresentor'):
                self.data_window = DataPresentor(self, True)
            self.data_window.show()
            self.hide()
        except Exception as e:
            print(e)

    def load_button_clicked(self):
        """ Loads a csv file with raw data and returns
            a pandas dataframe """
        f_name = QFileDialog.getOpenFileName(self, 'Open file')
        #self.data = pd.read_csv(f_name[0], engine='python')
        self.data = pd.read_excel(f_name[0])

class DataPresentor(QMainWindow):
    def __init__(self, parent, worst):
        super(DataPresentor, self).__init__(parent)
        self.worst = worst
        self.initUI()

    def initUI(self):
        self.setWindowIcon(QtGui.QIcon('Pirate.png'))
        self.setGeometry(screen_width/3, screen_height/2, screen_width/1.2, screen_height/1.5)
        self.setWindowTitle('Comment view')
        back_btn = QPushButton('Back', self)
        self.center()
        back_btn.clicked.connect(self.back_btn_clicked)
        back_btn.move(screen_width/1.5 + 100, 460)
        back_btn.resize(back_btn.sizeHint())
        approve_btn = QPushButton('Approve talkback', self)
        reject_btn = QPushButton('Reject talkback', self)
        approve_btn.clicked.connect(self.approve_btn_clicked)
        reject_btn.clicked.connect(self.reject_btn_clicked)
        approve_btn.move(screen_width/1.5 + 100, 100)
        reject_btn.move(screen_width/1.5 + 100, 180)
        approve_btn.resize(approve_btn.sizeHint())
        reject_btn.resize(reject_btn.sizeHint())

        df = self.parent().data.sort_values(by=['Prediction'], ascending=self.worst)[0:10]
        model = PandasModel(df)
        self.pandasTv = QtWidgets.QTableView(self)
        self.pandasTv.setModel(model)
        self.pandasTv.resize(screen_width/1.5, screen_height/2)
        # self.pandasTv.setFixedWidth(865)
        vbox = QVBoxLayout(self)
        vbox.addWidget(self.pandasTv)
        self.pandasTv.setSortingEnabled(True)

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def back_btn_clicked(self):
        try:
            self.parent().show()
            self.hide()
        except Exception as e:
            print(e)
        return

    def approve_btn_clicked(self):
        try:
            """ Drop first comment (will be added to site) """
            self.parent().data =self.parent().data.sort_values(by=['Prediction'], ascending=self.worst)[1:]
            # self.parent().data_window = DataPresentor(self.parent())
            self.hide()
            self.parent().present_btn_clicked()
        except Exception as e:
            print(e)
        return

    def reject_btn_clicked(self):
        try:
            """ Drop first comment (will be deleted from site) """
            self.parent().data =self.parent().data.sort_values(by=['Prediction'], ascending=self.worst)[1:]
            # self.parent().data_window = DataPresentor(self.parent())
            self.hide()
            self.parent().present_btn_clicked()
        except Exception as e:
            print(e)
        return


if __name__ == '__main__':
    app = QApplication(sys.argv)
    screen_height = app.primaryScreen().size().height()
    screen_width = app.primaryScreen().size().width()
    TalkBack_UI = TalkBackWindow()
    TalkBack_UI.show()
    app.aboutToQuit.connect(app.deleteLater)
    sys.exit(app.exec_())
