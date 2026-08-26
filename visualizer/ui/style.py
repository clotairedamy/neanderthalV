"""Dark application theme.

Applied once to the QApplication so dialogs inherit it too.
"""

BG = "#101116"          # window
PANEL = "#191b22"       # cards / group boxes
PANEL_HI = "#20232c"    # hover / inputs
BORDER = "#2a2e3a"
TEXT = "#e6e8ee"
MUTED = "#8b93a7"
ACCENT = "#8b5cf6"
ACCENT_HI = "#a78bfa"

STYLESHEET = f"""
QWidget {{
    background: {BG};
    color: {TEXT};
    font-family: "Helvetica Neue", "Segoe UI", Arial, sans-serif;
    font-size: 12px;
}}
QToolTip {{
    background: {PANEL_HI};
    color: {TEXT};
    border: 1px solid {BORDER};
    padding: 6px 8px;
    border-radius: 6px;
}}

/* ---- cards ---- */
QGroupBox {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 10px;
    margin-top: 16px;
    padding: 10px 10px 8px 10px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 4px;
    color: {MUTED};
    text-transform: uppercase;
    font-size: 10px;
    letter-spacing: 1px;
}}

/* ---- tabs ---- */
QTabWidget::pane {{
    border: none;
    background: {BG};
    top: -1px;
}}
QTabBar::tab {{
    background: transparent;
    color: {MUTED};
    padding: 8px 6px;
    margin-right: 1px;
    font-size: 11px;
    border: none;
    border-bottom: 2px solid transparent;
    font-weight: 600;
}}
QTabBar::tab:hover {{ color: {TEXT}; }}
QTabBar::tab:selected {{
    color: {TEXT};
    border-bottom: 2px solid {ACCENT};
}}

/* ---- buttons ---- */
QPushButton {{
    background: {PANEL_HI};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 7px 12px;
    color: {TEXT};
}}
QPushButton:hover {{ background: #272b36; border-color: #39405230; }}
QPushButton:pressed {{ background: #2f3442; }}
QPushButton:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
    color: #ffffff;
    font-weight: 600;
}}
QPushButton:disabled {{ color: {MUTED}; background: {PANEL}; }}

/* ---- inputs ---- */
QComboBox, QLineEdit, QDoubleSpinBox, QSpinBox {{
    background: {PANEL_HI};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 6px 8px;
    selection-background-color: {ACCENT};
}}
QComboBox:focus, QLineEdit:focus, QDoubleSpinBox:focus {{
    border-color: {ACCENT};
}}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{
    background: {PANEL_HI};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT};
    outline: none;
}}

/* ---- sliders ---- */
QSlider::groove:horizontal {{
    height: 4px;
    background: {BORDER};
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: #ffffff;
    width: 13px;
    height: 13px;
    margin: -5px 0;
    border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{ background: {ACCENT_HI}; }}
QSlider::groove:vertical {{
    width: 4px;
    background: {BORDER};
    border-radius: 2px;
}}
QSlider::sub-page:vertical {{ background: {BORDER}; }}
QSlider::add-page:vertical {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:vertical {{
    background: #ffffff;
    width: 13px;
    height: 13px;
    margin: 0 -5px;
    border-radius: 7px;
}}

/* ---- checks & radios ---- */
QCheckBox, QRadioButton {{ spacing: 8px; padding: 3px 0; }}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 15px; height: 15px;
    border: 1px solid #3a4152;
    background: {PANEL_HI};
}}
QCheckBox::indicator {{ border-radius: 4px; }}
QRadioButton::indicator {{ border-radius: 8px; }}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
}}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
    border-color: {ACCENT_HI};
}}

/* ---- lists & scroll ---- */
QListWidget {{
    background: {PANEL_HI};
    border: 1px solid {BORDER};
    border-radius: 8px;
    outline: none;
}}
QListWidget::item {{ padding: 4px 6px; border-radius: 4px; }}
QListWidget::item:selected {{ background: {ACCENT}; }}
QScrollArea {{ border: none; background: {BG}; }}
QScrollBar:vertical {{
    background: transparent; width: 10px; margin: 0;
}}
QScrollBar::handle:vertical {{
    background: #333846; border-radius: 5px; min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: #414859; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
QSplitter::handle {{ background: {BORDER}; width: 1px; }}
QProgressDialog {{ background: {PANEL}; }}
"""
