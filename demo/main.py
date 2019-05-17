import os
import wx
import wx.lib.agw.floatspin as FS
from glob import glob
import backend


class MainWindow(wx.Frame):
    def __init__(self, parent, title):
        wx.Frame.__init__(self, parent, title=title, size=(1248, 800))

        self.init_params()
        self.get_models()
        # self.pnl = wx.Panel(self)
        self.sizer = wx.BoxSizer(wx.VERTICAL)

        self.create_controls()

        self.bind_events()

        self.set_layout()

        self.Show(True)

    def init_params(self):
        self.models_dir = '../models/'

    def get_models(self):
        models = [i for i in os.walk(self.models_dir)][0][2]
        self.models = [
            i for i in models if not ('checkpoint' in i or 'csv' in i)
        ]

    def create_controls(self):

        self.load_sf_button = wx.Button(self, label='Load Soundfont')

        self.model_choice = wx.RadioBox(
            self, label='Model', choices=self.models, majorDimension=4)

        self.temperature = FS.FloatSpin(
            self,
            value=1.0,
            min_val=0.7,
            max_val=2.0,
            increment=0.1,
            style=wx.SL_HORIZONTAL | wx.SL_LABELS)
        self.temperature.SetFormat('%f')
        self.temperature.SetDigits(2)

        self.length = wx.Slider(
            self,
            value=300,
            minValue=100,
            maxValue=400,
            style=wx.SL_HORIZONTAL | wx.SL_LABELS)

        self.gen_button = wx.Button(self, label='Generate')

        self.play_button = wx.Button(self, label='Play Song')

        self.txt_section = wx.StaticText(
            self, label='Options', style=wx.ALIGN_CENTER)

        self.txt_temp = wx.StaticText(
            self, label='Temperature', style=wx.ALIGN_LEFT)

        self.txt_len = wx.StaticText(
            self, label='Song Length', style=wx.ALIGN_LEFT)

    def bind_events(self):
        self.Bind(wx.EVT_BUTTON, self.generate, self.gen_button)
        self.Bind(wx.EVT_BUTTON, self.load_soundfont, self.load_sf_button)
        self.Bind(wx.EVT_BUTTON, self.play_song, self.play_button)

    def load_soundfont(self, e):
        dlg = wx.FileDialog(self, 'Choose a Soundfont file', '~/', '', '*.*',
                            wx.FD_OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            fname = dlg.GetFilename()
            dirname = dlg.GetDirectory()
            self.soundfont = os.path.join(dirname, fname)

        dlg.Destroy()

    def set_layout(self):
        self.sizer.Add(self.load_sf_button, 0, wx.EXPAND)
        self.sizer.AddSpacer(10)
        self.sizer.Add(self.txt_section, 0, wx.EXPAND)
        self.sizer.Add(self.model_choice, 0, wx.EXPAND)
        self.sizer.AddSpacer(5)
        self.sizer.Add(self.txt_temp, 0, wx.EXPAND)
        self.sizer.Add(self.temperature, 0, wx.EXPAND)
        self.sizer.AddSpacer(5)
        self.sizer.Add(self.txt_len, 0, wx.EXPAND)
        self.sizer.Add(self.length, 0, wx.EXPAND)
        self.sizer.AddSpacer(5)
        self.sizer.Add(self.gen_button, 0, wx.EXPAND)
        self.sizer.Add(self.play_button, 0, wx.EXPAND)

        self.SetSizer(self.sizer)
        self.SetAutoLayout(1)
        self.sizer.Fit(self)

    def generate(self, e):
        choice = self.model_choice.GetSelection()
        backend.generate_song(
            self.models[choice],
            temperature=self.temperature.GetValue(),
            length=self.length.GetValue())

    def play_song(self, e):
        backend.play_song(self.soundfont)


if __name__ == '__main__':
    app = wx.App(False)
    frame = MainWindow(None, 'Demo Music Generator')
    app.MainLoop()
