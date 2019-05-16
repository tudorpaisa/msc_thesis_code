import os
import wx
from glob import glob


class MainWindow(wx.Frame):
    def __init__(self, parent, title):
        wx.Frame.__init__(self, parent, title=title, size=(1248, 800))

        self.init_params()
        self.get_models()
        self.create_controls()

        self.Show(True)

    def init_params(self):
        self.models_dir = '../models/'

    def get_models(self):
        models = [i for i in os.walk(self.models_dir)][0][2]
        self.models = [
            i for i in models if not ('checkpoint' in i or 'csv' in i)
        ]

    def create_controls(self):
        self.model_choice = wx.RadioBox(
            self,
            label='Choose a model',
            choices=self.models,
            majorDimension=4)
        self.gen_button = wx.Button(self, label='Generate')


if __name__ == '__main__':
    app = wx.App(False)
    frame = MainWindow(None, 'Demo Music Generator')
    app.MainLoop()
