"""Main window: preview on the left, simple controls on the right."""
import time
from pathlib import Path

from PySide6.QtCore import QThread, QTimer, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout,
                               QGroupBox, QHBoxLayout, QInputDialog, QLabel, QListWidget,
                               QListWidgetItem, QMainWindow, QMessageBox, QProgressDialog,
                               QPushButton, QRadioButton, QScrollArea, QSlider, QVBoxLayout,
                               QWidget)

from ..core.character import PROP_LAYERS, VIEW_LABELS, VIEW_NAMES, Action, Character, Prop
from ..core.project import load_project, save_project
from ..core.export import FORMATS, export_video
from ..core.imageio_utils import read_image
from ..core.keying import prepare_cutout
from ..core.scene import Scene
from ..core.skeleton import detect_joints
from .preview import PreviewWidget
from .qt_utils import thumbnail
from .rig_check import RigCheckDialog

IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
VIDEO_FILTER = "Videos (*.mp4 *.mov *.avi *.mkv *.webm *.m4v *.wmv)"
PROJECT_FILTER = "Puppet Animator project (*.puppet)"
ASSETS = Path(__file__).resolve().parents[2] / "assets"
# clips shown first, in this order; every other (non-hidden) clip follows
CLIP_ORDER = ["stop", "wave", "wave_other_arm", "jump", "point", "point_other_arm", "nod",
              "shake_head", "look_left", "look_right", "look_around", "clap", "shrug", "laugh",
              "celebrate", "dance", "sit", "stand", "drink", "drink_other_hand"]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Puppet Animator - prototype")
        self.scene = Scene()
        self.project_path = None
        self.playing = False
        self._last_tick = None

        self.preview = PreviewWidget(self.scene)
        self.preview.selectionChanged.connect(self._on_preview_select)
        self.preview.sceneEdited.connect(self._update_time_range)
        self.preview.targetPicked.connect(self._add_walk)

        # ---- top bar
        new_btn = QPushButton("New")
        new_btn.clicked.connect(self.new_project)
        open_btn = QPushButton("Open project...")
        open_btn.clicked.connect(self.open_project)
        save_btn = QPushButton("Save project")
        save_btn.clicked.connect(self.save_project)
        save_as_btn = QPushButton("Save as...")
        save_as_btn.clicked.connect(lambda: self.save_project(ask=True))
        bg_btn = QPushButton("Background image...")
        bg_btn.clicked.connect(self.load_background)
        bgv_btn = QPushButton("Background video...")
        bgv_btn.clicked.connect(self.load_background_video)
        add_btn = QPushButton("Add character...")
        add_btn.setStyleSheet("font-weight: bold;")
        add_btn.clicked.connect(self.add_character)
        export_btn = QPushButton("Export video...")
        export_btn.setStyleSheet("font-weight: bold;")
        export_btn.clicked.connect(self.export)
        top = QHBoxLayout()
        for b in (new_btn, open_btn, save_btn, save_as_btn, bg_btn, bgv_btn, add_btn):
            top.addWidget(b)
        top.addStretch()
        top.addWidget(export_btn)

        # ---- playback bar
        self.play_btn = QPushButton("▶  Play")
        self.play_btn.clicked.connect(self.toggle_play)
        rew = QPushButton("⏮  Start")
        rew.clicked.connect(lambda: self.set_time(0.0))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(lambda v: self.set_time(v / 100.0, from_slider=True))
        self.time_label = QLabel()
        self.time_label.setMinimumWidth(110)
        play = QHBoxLayout()
        for wdg in (rew, self.play_btn):
            play.addWidget(wdg)
        play.addWidget(self.slider, 1)
        play.addWidget(self.time_label)

        left = QVBoxLayout()
        left.addLayout(top)
        left.addWidget(self.preview, 1)
        left.addLayout(play)

        # ---- right panel: layers
        layers_box = QGroupBox("Characters (top of list = front)")
        self.layer_list = QListWidget()
        self.layer_list.setIconSize(self.layer_list.iconSize() * 2)
        self.layer_list.currentRowChanged.connect(self._on_list_select)
        self.layer_list.setMaximumHeight(110)
        fwd = QPushButton("Bring forward")
        fwd.clicked.connect(lambda: self.move_layer(+1))
        back = QPushButton("Send back")
        back.clicked.connect(lambda: self.move_layer(-1))
        flip = QPushButton("Flip")
        flip.clicked.connect(self.flip_selected)
        remove = QPushButton("Remove")
        remove.clicked.connect(self.remove_selected)
        lb = QGridLayout()
        lb.addWidget(fwd, 0, 0)
        lb.addWidget(back, 0, 1)
        lb.addWidget(flip, 1, 0)
        lb.addWidget(remove, 1, 1)
        add_view = QPushButton("Add view (side, back...)")
        add_view.clicked.connect(self.add_view)
        lb.addWidget(add_view, 2, 0)
        self.start_view = QComboBox()
        self.start_view.setToolTip("Which view the character starts in")
        self.start_view.activated.connect(self._set_start_view)
        lb.addWidget(self.start_view, 2, 1)

        # held objects
        self.prop_list = QListWidget()
        self.prop_list.setMaximumHeight(60)
        hold = QPushButton("Hold object...")
        hold.clicked.connect(lambda: self.add_prop(test=False))
        glass = QPushButton("Hold test glass")
        glass.clicked.connect(lambda: self.add_prop(test=True))
        edit_prop = QPushButton("Edit object")
        edit_prop.clicked.connect(self.edit_prop)
        del_prop = QPushButton("Remove object")
        del_prop.clicked.connect(self.remove_prop)
        hint = QLabel("Drag a character to move it. Mouse wheel over the picture = resize.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")
        lv = QVBoxLayout(layers_box)
        lv.addWidget(self.layer_list)
        lv.addLayout(lb)
        lv.addWidget(QLabel("Objects held:"))
        lv.addWidget(self.prop_list)
        pb = QGridLayout()
        pb.addWidget(hold, 0, 0)
        pb.addWidget(glass, 0, 1)
        pb.addWidget(edit_prop, 1, 0)
        pb.addWidget(del_prop, 1, 1)
        lv.addLayout(pb)
        lv.addWidget(hint)

        # ---- right panel: actions
        act_box = QGroupBox("What the selected character does (in order)")
        walk = QPushButton("Walk to...")
        walk.clicked.connect(self.start_walk_pick)
        idle = QPushButton("Stand still")
        self.idle_secs = QDoubleSpinBox()
        self.idle_secs.setRange(0.2, 60)
        self.idle_secs.setValue(2.0)
        self.idle_secs.setSuffix(" s")
        idle.clicked.connect(lambda: self.add_action(Action("idle", duration=self.idle_secs.value())))
        grid = QGridLayout()
        grid.addWidget(walk, 0, 0)
        idle_row = QHBoxLayout()
        idle_row.addWidget(idle)
        idle_row.addWidget(self.idle_secs)
        grid.addLayout(idle_row, 0, 1)
        turn = QPushButton("Turn to view...")
        turn.clicked.connect(self.add_turn)
        grid.addWidget(turn, 1, 0)
        # every animation file in the library gets a button (new files too)
        lib = self.scene.lib.clips
        clips = [c for c in CLIP_ORDER if c in lib] + sorted(
            c for c in lib if c not in CLIP_ORDER and c not in ("idle", "walk") and not lib[c].hidden)
        clip_grid = QGridLayout()
        for i, clip in enumerate(clips):
            b = QPushButton(lib[clip].name.replace(" (", "\n("))
            b.setToolTip(lib[clip].name)
            b.clicked.connect(lambda _=False, c=clip: self.add_action(Action(c)))
            clip_grid.addWidget(b, i // 3, i % 3)
        # the step list comes first so you can always see what's been added
        self.action_list = QListWidget()
        self.action_list.setMinimumHeight(170)
        self.action_list.itemClicked.connect(self._jump_to_step)
        QShortcut(QKeySequence.Delete, self.action_list, activated=self.remove_action)
        steps_hint = QLabel("Click a step to jump to it. The highlighted step is the one playing.")
        steps_hint.setWordWrap(True)
        steps_hint.setStyleSheet("color: gray;")
        del_act = QPushButton("Remove selected")
        del_act.clicked.connect(self.remove_action)
        undo = QPushButton("Undo last")
        undo.clicked.connect(self.undo_last_action)
        up = QPushButton("Move up")
        up.clicked.connect(lambda: self.move_action(-1))
        down = QPushButton("Move down")
        down.clicked.connect(lambda: self.move_action(+1))
        clear = QPushButton("Clear all")
        clear.clicked.connect(self.clear_actions)
        ab = QGridLayout()
        ab.addWidget(del_act, 0, 0)
        ab.addWidget(undo, 0, 1)
        ab.addWidget(clear, 0, 2)
        ab.addWidget(up, 1, 0)
        ab.addWidget(down, 1, 1)
        add_label = QLabel("<b>Add a step:</b>")
        av = QVBoxLayout(act_box)
        av.addWidget(self.action_list)
        av.addWidget(steps_hint)
        av.addLayout(ab)
        av.addWidget(add_label)
        av.addLayout(grid)
        av.addLayout(clip_grid)
        self.act_box = act_box

        right = QVBoxLayout()
        right.addWidget(layers_box)
        right.addWidget(act_box, 1)
        right_inner = QWidget()
        right_inner.setLayout(right)
        right_w = QScrollArea()
        right_w.setWidget(right_inner)
        right_w.setWidgetResizable(True)
        right_w.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_w.setFixedWidth(470)

        root = QHBoxLayout()
        root.addLayout(left, 1)
        root.addWidget(right_w)
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)
        self.resize(1500, 900)

        self.timer = QTimer(self)
        self.timer.setInterval(15)
        self.timer.timeout.connect(self._tick)
        self._refresh_lists()
        self._update_time_range()
        self.preview.refresh()

    # ------------------------------------------------------------ helpers
    @property
    def selected(self):
        return self.preview.selected

    def _select(self, ch):
        self.preview.selected = ch
        self._refresh_lists()
        self.preview.update()

    def _on_preview_select(self, ch):
        self._refresh_lists()

    def _on_list_select(self, row):
        if row < 0:
            return
        idx = len(self.scene.characters) - 1 - row
        ch = self.scene.characters[idx]
        if ch is not self.preview.selected:
            self.preview.selected = ch
            self._refresh_actions()
            self.preview.update()

    def _refresh_lists(self):
        self.layer_list.blockSignals(True)
        self.layer_list.clear()
        for ch in reversed(self.scene.characters):
            item = QListWidgetItem(QIcon(thumbnail(ch.rig.image, 64)), ch.name)
            self.layer_list.addItem(item)
            if ch is self.selected:
                self.layer_list.setCurrentItem(item)
        self.layer_list.blockSignals(False)
        self._refresh_actions()

    def _refresh_character_extras(self):
        ch = self.selected
        self.start_view.blockSignals(True)
        self.start_view.clear()
        self.prop_list.clear()
        if ch:
            for v in VIEW_NAMES:
                if v in ch.views:
                    self.start_view.addItem(f"Starts: {VIEW_LABELS[v]}", v)
            self.start_view.setCurrentIndex(max(self.start_view.findData(ch.start_view), 0))
            for pr in ch.props:
                side = "left" if pr.hand == "l" else "right"
                self.prop_list.addItem(f"{pr.name} - hand on {side} of picture - {PROP_LAYERS[pr.layer]}")
        self.start_view.blockSignals(False)

    def _refresh_actions(self):
        self._refresh_character_extras()
        self.action_list.clear()
        ch = self.selected
        self.act_box.setEnabled(ch is not None)
        self.act_box.setTitle(f"Steps for {ch.name} (in order)" if ch else
                              "Select a character to see and add its steps")
        if ch:
            times = ch.action_times(self.scene.lib)
            for i, (a, tt) in enumerate(zip(ch.actions, times)):
                when = f"{tt[0]:5.1f}-{tt[1]:5.1f}s" if tt else "  (skipped)  "
                self.action_list.addItem(f"{i + 1}.  {when}   {a.label(self.scene.lib)}")
            if not ch.actions:
                self.action_list.addItem("(no steps yet - add some below)")
            self.action_list.addItem("then rests until the end")
        self._highlight_step()

    def _highlight_step(self):
        """Mark the step that is playing at the current time."""
        ch = self.selected
        if ch is None:
            return
        cur = ch.action_at(self.preview.time, self.scene.lib)
        last = self.action_list.count() - 1
        for i in range(self.action_list.count()):
            item = self.action_list.item(i)
            active = (i == cur) or (cur < 0 and i == last)
            item.setBackground(QBrush(QColor(255, 236, 150)) if active else QBrush())
            f = item.font()
            f.setBold(active)
            item.setFont(f)

    def _jump_to_step(self, item):
        ch = self.selected
        row = self.action_list.row(item)
        if ch and 0 <= row < len(ch.actions):
            tt = ch.action_times(self.scene.lib)[row]
            if tt:
                self.set_time(tt[0] + 0.01)

    def _update_time_range(self):
        d = self.scene.duration()
        self.slider.blockSignals(True)
        self.slider.setRange(0, int(d * 100))
        self.slider.blockSignals(False)
        self._show_time()

    def _show_time(self):
        self.time_label.setText(f"{self.preview.time:5.2f} / {self.scene.duration():.2f} s")
        self._highlight_step()

    def _changed(self):
        self._refresh_actions()
        self._update_time_range()
        self.preview.refresh()

    # ------------------------------------------------------------ playback
    def set_time(self, t, from_slider=False):
        self.preview.time = max(0.0, t)
        if not from_slider:
            self.slider.blockSignals(True)
            self.slider.setValue(int(t * 100))
            self.slider.blockSignals(False)
        self._show_time()
        self.preview.refresh()

    def toggle_play(self):
        self.playing = not self.playing
        self.play_btn.setText("⏸  Pause" if self.playing else "▶  Play")
        if self.playing:
            if self.preview.time >= self.scene.duration() - 0.05:
                self.set_time(0.0)
            self._last_tick = time.perf_counter()
            self.timer.start()
        else:
            self.timer.stop()

    def _tick(self):
        now = time.perf_counter()
        t = self.preview.time + (now - self._last_tick)
        self._last_tick = now
        if t >= self.scene.duration():
            t = self.scene.duration()
            self.toggle_play()
        self.set_time(t)

    # ------------------------------------------------------------ actions
    def load_background(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose background image", "", IMAGE_FILTER)
        if not path:
            return
        try:
            self.scene.set_background(read_image(path))
        except Exception as e:
            QMessageBox.warning(self, "Background", str(e))
            return
        self.preview.refresh()

    def add_character(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose character image (plain-colour background)", "", IMAGE_FILTER)
        if not path:
            return
        rig = self._make_rig(path, "Add character")
        if rig is None:
            return
        n = len(self.scene.characters)
        ch = Character(rig, pos=(0.3 + 0.2 * (n % 3), 0.92), height=0.6)
        self.scene.characters.append(ch)
        self._select(ch)
        self._changed()

    def move_layer(self, d):
        ch = self.selected
        if ch is None:
            return
        cs = self.scene.characters
        i = cs.index(ch)
        j = min(max(i + d, 0), len(cs) - 1)
        cs.insert(j, cs.pop(i))
        self._refresh_lists()
        self.preview.refresh()

    def flip_selected(self):
        if self.selected:
            self.selected.mirrored = not self.selected.mirrored
            self.preview.refresh()

    def remove_selected(self):
        ch = self.selected
        if ch is None:
            return
        self.scene.characters.remove(ch)
        self.preview.selected = None
        self._refresh_lists()
        self._changed()

    def start_walk_pick(self):
        if self.selected is None:
            return
        self.preview.picking_target = True
        self.preview.setFocus()
        self.preview.update()

    def _add_walk(self, x, y):
        if self.selected is None:
            return
        self.selected.actions.append(Action("walk", target=(x, y)))
        self.selected.actions.append(Action("stop"))
        self._changed()

    def add_action(self, action):
        if self.selected is None:
            return
        self.selected.actions.append(action)
        self._changed()

    def remove_action(self):
        ch = self.selected
        row = self.action_list.currentRow()
        if ch and 0 <= row < len(ch.actions):
            del ch.actions[row]
            self._changed()
            self.action_list.setCurrentRow(min(row, len(ch.actions) - 1))
        elif ch and ch.actions:
            QMessageBox.information(self, "Remove step", "Click a step in the list first.")

    def undo_last_action(self):
        if self.selected and self.selected.actions:
            self.selected.actions.pop()
            self._changed()

    def move_action(self, d):
        ch = self.selected
        row = self.action_list.currentRow()
        if ch and 0 <= row < len(ch.actions) and 0 <= row + d < len(ch.actions):
            ch.actions.insert(row + d, ch.actions.pop(row))
            self._changed()
            self.action_list.setCurrentRow(row + d)

    def clear_actions(self):
        if self.selected:
            self.selected.actions.clear()
            self._changed()

    # ------------------------------------------------------------ project
    def _set_scene(self, scene):
        if self.playing:
            self.toggle_play()
        self.scene = scene
        self.preview.scene = scene
        self.preview.selected = None
        self.set_time(0.0)
        self._refresh_lists()
        self._changed()

    def new_project(self):
        if self.scene.characters and QMessageBox.question(
                self, "New project", "Start a new empty project? Unsaved changes are lost.") != QMessageBox.Yes:
            return
        self.project_path = None
        self.setWindowTitle("Puppet Animator - prototype")
        self._set_scene(Scene())

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open project", "", PROJECT_FILTER)
        if not path:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            scene = load_project(path)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Open project", f"Could not open the project:\n{e}")
            return
        QApplication.restoreOverrideCursor()
        self.project_path = path
        self.setWindowTitle(f"Puppet Animator - {Path(path).name}")
        self._set_scene(scene)

    def save_project(self, ask=False):
        path = self.project_path
        if ask or not path:
            path, _ = QFileDialog.getSaveFileName(self, "Save project", path or "my animation.puppet", PROJECT_FILTER)
            if not path:
                return
            if not path.lower().endswith(".puppet"):
                path += ".puppet"
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            save_project(self.scene, path)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Save project", f"Could not save:\n{e}")
            return
        QApplication.restoreOverrideCursor()
        self.project_path = path
        self.setWindowTitle(f"Puppet Animator - {Path(path).name}")

    def load_background_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose background video", "", VIDEO_FILTER)
        if not path:
            return
        try:
            self.scene.set_background_video(path)
        except Exception as e:
            QMessageBox.warning(self, "Background video", str(e))
            return
        self.preview.refresh()

    # ------------------------------------------------------------ views
    def _make_rig(self, path, title):
        """Background removal + joint detection + 'check the dots'. Returns Rig or None."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            cut = prepare_cutout(read_image(path))
            joints, method = detect_joints(cut)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, title, f"Could not process this image:\n{e}")
            return None
        QApplication.restoreOverrideCursor()
        dlg = RigCheckDialog(cut, joints, method, Path(path).stem, self)
        return dlg.rig if dlg.exec() == QDialog.Accepted else None

    def add_view(self):
        ch = self.selected
        if ch is None:
            return
        labels = [VIEW_LABELS[v] for v in VIEW_NAMES if v != "front"]
        label, ok = QInputDialog.getItem(self, "Add view", "Which view is this picture?", labels, 1, False)
        if not ok:
            return
        view = next(v for v in VIEW_NAMES if VIEW_LABELS[v] == label)
        path, _ = QFileDialog.getOpenFileName(self, f"Choose the {label} picture of {ch.name}", "", IMAGE_FILTER)
        if not path:
            return
        rig = self._make_rig(path, "Add view")
        if rig is None:
            return
        if view != "back":
            rig.faces_left = QMessageBox.question(
                self, "Add view", "Is the character facing LEFT in this picture?\n"
                "(Walking direction is worked out from this.)") == QMessageBox.Yes
        ch.views[view] = rig
        self._changed()

    def _set_start_view(self, i):
        if self.selected:
            self.selected.start_view = self.start_view.itemData(i)
            self._changed()

    def add_turn(self):
        ch = self.selected
        if ch is None:
            return
        views = [v for v in VIEW_NAMES if v in ch.views]
        if len(views) < 2:
            QMessageBox.information(self, "Turn", "This character only has one view.\n"
                                    "Use \u201cAdd view\u201d to give it a side/back picture first.")
            return
        label, ok = QInputDialog.getItem(self, "Turn", "Turn to which view?", [VIEW_LABELS[v] for v in views], 0, False)
        if ok:
            self.add_action(Action("turn", view=next(v for v in views if VIEW_LABELS[v] == label)))

    # ------------------------------------------------------------ props
    def add_prop(self, test=False):
        ch = self.selected
        if ch is None:
            return
        if test:
            img, name = read_image(ASSETS / "props" / "glass.png"), "Glass"
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Choose object picture (plain background or transparent)", "", IMAGE_FILTER)
            if not path:
                return
            try:
                img, name = prepare_cutout(read_image(path)), Path(path).stem
            except Exception as e:
                QMessageBox.warning(self, "Hold object", str(e))
                return
        prop = Prop(img, name)
        if PropDialog(prop, self).exec() == QDialog.Accepted:
            ch.props.append(prop)
            self._changed()

    def edit_prop(self):
        ch, row = self.selected, self.prop_list.currentRow()
        if ch and 0 <= row < len(ch.props):
            PropDialog(ch.props[row], self).exec()
            self._changed()

    def remove_prop(self):
        ch, row = self.selected, self.prop_list.currentRow()
        if ch and 0 <= row < len(ch.props):
            del ch.props[row]
            self._changed()

    # ------------------------------------------------------------ export
    def export(self):
        if not self.scene.characters:
            QMessageBox.information(self, "Export", "Add a character first.")
            return
        if self.playing:
            self.toggle_play()
        dlg = ExportDialog(self.scene, self.selected, self)
        if dlg.exec() != QDialog.Accepted:
            return
        jobs = dlg.jobs()
        if not jobs:
            return
        progress = QProgressDialog("Exporting...", "Cancel", 0, 1000, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        worker = ExportWorker(self.scene, jobs, dlg.size(), dlg.duration.value())
        worker.progressed.connect(lambda v: progress.setValue(int(v * 1000)))
        worker.label.connect(progress.setLabelText)
        progress.canceled.connect(worker.cancel)
        worker.done.connect(lambda msg: (progress.reset(), QMessageBox.information(self, "Export", msg)))
        self._worker = worker
        worker.start()


class ExportWorker(QThread):
    progressed = Signal(float)
    label = Signal(str)
    done = Signal(str)

    def __init__(self, scene, jobs, size, duration):
        super().__init__()
        self.scene, self.jobs, self.size, self.duration = scene, jobs, size, duration
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        written = []
        try:
            for k, (path, fmt, with_bg, only) in enumerate(self.jobs):
                self.label.emit(f"Exporting {Path(path).name} ({k + 1} of {len(self.jobs)})...")
                out = export_video(self.scene, path, fmt, with_bg, only, *self.size, fps=30,
                                   duration=self.duration,
                                   progress=lambda f, k=k: self.progressed.emit((k + f) / len(self.jobs)),
                                   cancelled=lambda: self._cancel)
                if out is None:
                    self.done.emit("Export cancelled.")
                    return
                written.append(str(out))
        except Exception as e:
            self.done.emit(f"Export failed:\n{e}")
            return
        self.done.emit("Saved:\n" + "\n".join(written))


class ExportDialog(QDialog):
    def __init__(self, scene, selected, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export video")
        self.scene, self.selected = scene, selected
        self.r_scene = QRadioButton("Whole scene with background (MP4)")
        self.r_chars = QRadioButton("All characters together, transparent background")
        self.r_each = QRadioButton("Each character as its own transparent video")
        self.r_each.setChecked(True)
        self.fmt = QComboBox()
        for key in ("mov", "webm", "png", "green"):
            self.fmt.addItem(FORMATS[key][0], key)
        self.res = QComboBox()
        self.res.addItem("1920 x 1080 (Full HD)", (1920, 1080))
        self.res.addItem("1280 x 720", (1280, 720))
        self.duration = QDoubleSpinBox()
        self.duration.setRange(0.5, 600)
        self.duration.setDecimals(1)
        self.duration.setValue(round(scene.duration(), 1))
        self.duration.setSuffix(" s")
        self.r_scene.toggled.connect(lambda on: self.fmt.setEnabled(not on))
        form = QFormLayout()
        form.addRow("Transparent format:", self.fmt)
        form.addRow("Size:", self.res)
        form.addRow("Length:", self.duration)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        for w in (self.r_scene, self.r_chars, self.r_each):
            lay.addWidget(w)
        lay.addLayout(form)
        lay.addWidget(bb)

    def size(self):
        return self.res.currentData()

    def jobs(self):
        """Ask where to save; returns list of (path, fmt, with_background, only)."""
        if self.r_scene.isChecked():
            path, _ = QFileDialog.getSaveFileName(self, "Save scene video", "scene.mp4", "MP4 video (*.mp4)")
            return [(path, "mp4", True, None)] if path else []
        fmt = self.fmt.currentData()
        ext = FORMATS[fmt][1]
        folder = QFileDialog.getExistingDirectory(self, "Choose a folder to save into")
        if not folder:
            return []
        if self.r_chars.isChecked():
            return [(str(Path(folder) / f"characters{ext}"), fmt, False, None)]
        jobs, used = [], set()
        for ch in self.scene.characters:
            base = "".join(c for c in ch.name if c.isalnum() or c in " _-") or "character"
            name, n = base, 2
            while name in used:
                name, n = f"{base}_{n}", n + 1
            used.add(name)
            jobs.append((str(Path(folder) / f"{name}{ext}"), fmt, False, ch))
        return jobs


class PropDialog(QDialog):
    """Settings for a held object. Changes apply immediately to the object."""

    def __init__(self, prop, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Held object: {prop.name}")
        self.prop = prop
        self.hand = QComboBox()
        self.hand.addItem("Hand on the LEFT of the picture", "l")
        self.hand.addItem("Hand on the RIGHT of the picture", "r")
        self.hand.setCurrentIndex(0 if prop.hand == "l" else 1)
        self.size = self._spin(0.02, 1.0, prop.size, 0.01)
        self.dx = self._spin(-0.5, 0.5, prop.offset[0], 0.01)
        self.dy = self._spin(-0.5, 0.5, prop.offset[1], 0.01)
        self.rot = self._spin(-180, 180, prop.rotation, 5)
        self.follow = self._spin(0, 1, prop.follow, 0.1)
        self.layer = QComboBox()
        self.layer.addItems(PROP_LAYERS)
        self.layer.setCurrentIndex(prop.layer)
        form = QFormLayout()
        form.addRow("Held in:", self.hand)
        form.addRow("Size (x character height):", self.size)
        form.addRow("Move right/left:", self.dx)
        form.addRow("Move down/up:", self.dy)
        form.addRow("Rotate (degrees):", self.rot)
        form.addRow("Turns with the arm (0-1):", self.follow)
        form.addRow("Draw:", self.layer)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._apply)
        bb.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(bb)

    @staticmethod
    def _spin(lo, hi, val, step):
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setDecimals(2)
        sp.setSingleStep(step)
        sp.setValue(val)
        return sp

    def _apply(self):
        p = self.prop
        p.hand = self.hand.currentData()
        p.size = self.size.value()
        p.offset = (self.dx.value(), self.dy.value())
        p.rotation = self.rot.value()
        p.follow = self.follow.value()
        p.layer = self.layer.currentIndex()
        self.accept()
