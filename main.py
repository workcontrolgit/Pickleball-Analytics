"""
Tkinter desktop UI (customtkinter) for the video analytics pipeline

Purpose
-------
Provides a modern, sporty card-based GUI to pick a video, configure processing
mode, launch processing on a background thread, show progress, and display
analytics results.

What it does
------------
- Select video (file dialog) → enables mode selector and "Process Video"
- Spawns VideoProcessor(video_path, filters) on a thread; wires a progress callback
- Updates a progress bar and status label
- Shows analytics badges and a results card with rally count and output path

Primary entry point
-------------------
- App.process_video(): prepares always-true filters and calls VideoProcessor.process_video()
- __main__: launches the Tkinter event loop
"""

import os
import customtkinter as ctk
from tkinter import filedialog
from process_video import (
    VideoProcessor,
    MODE_VIDEO_ANALYSIS,
    MODE_SPLIT_RALLIES,
    MODE_DETECT_SERVE,
)
import threading

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

# ── Palette ──────────────────────────────────────────────────────────────────
BG          = "#0F1117"
CARD_BG     = "#1A1D27"
CARD_BORDER = "#2A2D3A"
CTA         = "#00D4FF"
RALLY_STAT  = "#FF6B35"
SUCCESS     = "#00C49A"
ERROR       = "#FF4757"
BODY_TEXT   = "#E0E0E0"
MUTED_TEXT  = "#6B7280"


MODE_DESCRIPTIONS = {
    "Video Analysis": "Produces an annotated video with overlays and analytics",
    "Split Rallies":  "Finds long rallies and saves each one as a clip",
    "Detect Serve":   "Scores serves using AI vision — fastest mode",
}


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Pickleball Analytics")
        self.geometry("900x600")
        self.resizable(False, False)
        self.configure(fg_color=BG)

        self.video_path = None
        self.out_dir = None
        self.mode_var = ctk.StringVar(value="Video Analysis")
        self._reveal_id = None

        self._build_ui()
        self._set_state_idle()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # Three rows: header / cards / footer
        self.grid_rowconfigure(0, weight=0, minsize=50)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0, minsize=60)
        self.grid_columnconfigure(0, weight=1)

        self._build_header()
        self._build_card_area()
        self._build_footer()

    # Zone 1 — Header Strip
    def _build_header(self):
        header = ctk.CTkFrame(self, fg_color=BG, corner_radius=0, height=50)
        header.grid(row=0, column=0, sticky="ew", padx=20)
        header.grid_propagate(False)
        header.grid_columnconfigure(0, weight=1)
        header.grid_columnconfigure(1, weight=0)

        ctk.CTkLabel(
            header,
            text="Pickleball Analytics",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="white",
        ).grid(row=0, column=0, sticky="w", pady=10)

        # Filename chip (hidden until file selected)
        self.chip_frame = ctk.CTkFrame(
            header, fg_color=CARD_BORDER, corner_radius=12, height=28
        )
        self.chip_label = ctk.CTkLabel(
            self.chip_frame,
            text="",
            font=ctk.CTkFont(size=12),
            text_color="white",
        )
        self.chip_label.pack(padx=12, pady=4)
        # Initially hidden
        self.chip_frame.grid(row=0, column=1, sticky="e", pady=10)
        self.chip_frame.grid_remove()

    # Zone 2 — Main Card Area (2×2 grid)
    def _build_card_area(self):
        area = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        area.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 10))
        area.grid_columnconfigure(0, weight=1)
        area.grid_columnconfigure(1, weight=1)
        area.grid_rowconfigure(0, weight=1)
        area.grid_rowconfigure(1, weight=1)

        self._build_file_card(area)
        self._build_mode_card(area)
        self._build_results_card(area)

    def _make_card(self, parent, row, col, **kwargs):
        defaults = dict(
            fg_color=CARD_BG,
            border_color=CARD_BORDER,
            border_width=1,
            corner_radius=12,
        )
        defaults.update(kwargs)
        card = ctk.CTkFrame(parent, **defaults)
        card.grid(row=row, column=col, padx=8, pady=8, sticky="nsew")
        return card

    # File Card
    def _build_file_card(self, parent):
        card = self._make_card(parent, 0, 0)
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(0, weight=1)
        card.grid_rowconfigure(1, weight=0)

        # Placeholder shown before a file is chosen
        self.drop_label = ctk.CTkLabel(
            card,
            text="Drop or browse for\na video file",
            font=ctk.CTkFont(size=14),
            text_color=MUTED_TEXT,
            justify="center",
        )
        self.drop_label.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="nsew")

        # Info block shown after a file is chosen (hidden initially)
        self.file_info_frame = ctk.CTkFrame(card, fg_color="transparent")
        self.file_name_label    = ctk.CTkLabel(self.file_info_frame, text="", anchor="w",
                                               font=ctk.CTkFont(size=13, weight="bold"),
                                               text_color=BODY_TEXT, wraplength=260)
        self.file_folder_label  = ctk.CTkLabel(self.file_info_frame, text="", anchor="w",
                                               font=ctk.CTkFont(size=11),
                                               text_color=MUTED_TEXT, wraplength=260)
        self.file_date_label    = ctk.CTkLabel(self.file_info_frame, text="", anchor="w",
                                               font=ctk.CTkFont(size=11),
                                               text_color=MUTED_TEXT)
        self.file_size_label    = ctk.CTkLabel(self.file_info_frame, text="", anchor="w",
                                               font=ctk.CTkFont(size=11),
                                               text_color=MUTED_TEXT)
        self.file_name_label.pack(anchor="w", pady=(0, 2))
        self.file_folder_label.pack(anchor="w")
        self.file_date_label.pack(anchor="w")
        self.file_size_label.pack(anchor="w")
        # hidden until a file is selected
        self.file_info_frame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="nsew")
        self.file_info_frame.grid_remove()

        self.browse_btn = ctk.CTkButton(
            card,
            text="Browse Video",
            fg_color=CTA,
            text_color="black",
            font=ctk.CTkFont(size=13, weight="bold"),
            corner_radius=8,
            command=self.select_video,
        )
        self.browse_btn.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="ew")

    # Mode Card
    def _build_mode_card(self, parent):
        card = self._make_card(parent, 0, 1)
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(0, weight=0)
        card.grid_rowconfigure(1, weight=0)
        card.grid_rowconfigure(2, weight=0)
        card.grid_rowconfigure(3, weight=1)
        card.grid_rowconfigure(4, weight=0)

        ctk.CTkLabel(
            card,
            text="Processing Mode",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=BODY_TEXT,
        ).grid(row=0, column=0, padx=16, pady=(16, 6), sticky="w")

        self.mode_selector = ctk.CTkSegmentedButton(
            card,
            values=["Video Analysis", "Split Rallies", "Detect Serve"],
            variable=self.mode_var,
            state="disabled",
            selected_color=CTA,
            selected_hover_color="#00B8D9",
            unselected_color=CARD_BORDER,
            font=ctk.CTkFont(size=12),
            command=self._on_mode_changed,
        )
        self.mode_selector.grid(row=1, column=0, padx=16, pady=8, sticky="ew")

        self.mode_desc_label = ctk.CTkLabel(
            card,
            text=MODE_DESCRIPTIONS.get(self.mode_var.get(), ""),
            font=ctk.CTkFont(size=11),
            text_color=MUTED_TEXT,
            wraplength=280,
            justify="left",
        )
        self.mode_desc_label.grid(row=2, column=0, padx=16, pady=(0, 8), sticky="w")

        self.process_btn = ctk.CTkButton(
            card,
            text="Process Video",
            fg_color=CTA,
            text_color="black",
            font=ctk.CTkFont(size=13, weight="bold"),
            corner_radius=8,
            state="disabled",
            command=self.process_video_thread,
        )
        self.process_btn.grid(row=4, column=0, padx=16, pady=(0, 16), sticky="ew")

    # Results Card
    def _build_results_card(self, parent):
        self.results_card = self._make_card(parent, 1, 0, height=0)
        self.results_card.grid(
            row=1, column=0, columnspan=2, padx=8, pady=8, sticky="nsew"
        )
        self.results_card.grid_propagate(False)
        self.results_card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self.results_card,
            text="Results",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=BODY_TEXT,
        ).grid(row=0, column=0, padx=16, pady=(12, 4), sticky="w")

        # Rally count badge
        self.rally_badge_frame = ctk.CTkFrame(
            self.results_card, fg_color=RALLY_STAT, corner_radius=12
        )
        self.rally_badge_label = ctk.CTkLabel(
            self.rally_badge_frame,
            text="",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="white",
        )
        self.rally_badge_label.pack(padx=12, pady=6)
        self.rally_badge_frame.grid(row=1, column=0, padx=16, pady=4, sticky="w")

        # Output path
        self.output_path_label = ctk.CTkLabel(
            self.results_card,
            text="",
            font=ctk.CTkFont(size=11),
            text_color=MUTED_TEXT,
            wraplength=300,
            justify="left",
        )
        self.output_path_label.grid(row=2, column=0, padx=16, pady=(0, 4), sticky="w")

        # Open folder button
        self.open_folder_btn = ctk.CTkButton(
            self.results_card,
            text="📂 Open Folder",
            fg_color=CARD_BORDER,
            text_color=BODY_TEXT,
            font=ctk.CTkFont(size=12),
            corner_radius=8,
            command=self._open_output_folder,
        )
        self.open_folder_btn.grid(row=3, column=0, padx=16, pady=(0, 12), sticky="w")

    # Zone 3 — Footer
    def _build_footer(self):
        footer = ctk.CTkFrame(self, fg_color=BG, corner_radius=0, height=60)
        footer.grid(row=2, column=0, sticky="ew", padx=20)
        footer.grid_propagate(False)
        footer.grid_columnconfigure(0, weight=1)

        self.progress_bar = ctk.CTkProgressBar(
            footer,
            fg_color=CARD_BORDER,
            progress_color=CTA,
            height=8,
            corner_radius=4,
        )
        self.progress_bar.set(0)
        self.progress_bar.grid(row=0, column=0, sticky="ew", pady=(8, 2))

        self.status_label = ctk.CTkLabel(
            footer,
            text="",
            font=ctk.CTkFont(size=12),
            text_color=MUTED_TEXT,
            justify="left",
            anchor="w",
        )
        self.status_label.grid(row=1, column=0, sticky="w")

    # ── State machine ─────────────────────────────────────────────────────────

    def _set_state_idle(self):
        self.mode_selector.configure(state="disabled")
        self.process_btn.configure(state="disabled")
        self.chip_frame.grid_remove()
        self.file_info_frame.grid_remove()   # hide info block
        self.drop_label.grid()               # show placeholder again
        self.progress_bar.set(0)
        self.status_label.configure(
            text="Select a video to begin", text_color=MUTED_TEXT
        )

    def _on_mode_changed(self, value: str) -> None:
        """Update the description label when the user picks a different mode."""
        self.mode_desc_label.configure(text=MODE_DESCRIPTIONS.get(value, ""))

    def _set_state_file_selected(self):
        import os
        from datetime import datetime

        path = self.video_path
        filename   = os.path.basename(path)
        folder     = os.path.dirname(path)
        size_mb    = os.path.getsize(path) / (1024 * 1024)
        modified   = datetime.fromtimestamp(os.path.getmtime(path))
        date_str   = modified.strftime("%Y-%m-%d  %H:%M")

        # Update chip in header
        self.chip_label.configure(text=filename)
        self.chip_frame.grid()

        # Fill info labels
        self.file_name_label.configure(text=filename)
        self.file_folder_label.configure(text=folder)
        self.file_date_label.configure(text=f"Modified: {date_str}")
        self.file_size_label.configure(text=f"Size: {size_mb:.1f} MB")

        # Swap placeholder for info block
        self.drop_label.grid_remove()
        self.file_info_frame.grid()

        self.mode_selector.configure(state="normal")
        self.process_btn.configure(state="normal")
        self.status_label.configure(text="Ready to process", text_color=BODY_TEXT)

    def _set_state_processing(self):
        self.browse_btn.configure(state="disabled")
        self.mode_selector.configure(state="disabled")
        self.process_btn.configure(state="disabled")
        self.progress_bar.set(0)
        self.status_label.configure(text="Analyzing\u2026", text_color=MUTED_TEXT)

    def _set_state_complete(self, rally_count, serve_count, serve_avg, out_dir, mode):
        self.browse_btn.configure(state="normal")
        self.mode_selector.configure(state="normal")
        self.process_btn.configure(state="normal")
        self.progress_bar.set(1.0)
        self.status_label.configure(text="Done!", text_color=SUCCESS)

        if mode == MODE_VIDEO_ANALYSIS:
            badge_text = "Output saved"
        elif mode == MODE_SPLIT_RALLIES:
            badge_text = (
                f"{rally_count} long {'rally' if rally_count == 1 else 'rallies'} found"
                if rally_count > 0
                else "No long rallies detected"
            )
        elif mode == MODE_DETECT_SERVE:
            badge_text = (
                (
                    f"{serve_count} {'serve' if serve_count == 1 else 'serves'} detected"
                    f" · avg score {serve_avg}/10"
                )
                if serve_count > 0
                else "No serves detected"
            )
        else:
            badge_text = "Processing complete"

        self.rally_badge_label.configure(text=badge_text)
        self.output_path_label.configure(text=out_dir)
        self.out_dir = out_dir
        self._reveal_results()

    def _set_state_error(self, error_msg):
        self.browse_btn.configure(state="normal")
        self.mode_selector.configure(state="normal")
        self.process_btn.configure(state="normal")
        self.status_label.configure(
            text=f"Error: {error_msg}", text_color=ERROR
        )

    # ── Results card reveal animation ─────────────────────────────────────────

    def _reveal_results(self, step=0):
        target = 90
        increment = 10
        delay = 30
        current = step * increment
        if current <= target:
            self.results_card.configure(height=current)
            self._reveal_id = self.after(delay, lambda: self._reveal_results(step + 1))
        else:
            self._reveal_id = None

    # ── Actions ───────────────────────────────────────────────────────────────

    def select_video(self):
        filetypes = (("MP4 files", "*.mp4"), ("All files", "*.*"))
        path = filedialog.askopenfilename(
            title="Open video file", filetypes=filetypes
        )
        if path:
            self.video_path = path
            if self._reveal_id is not None:
                self.after_cancel(self._reveal_id)
                self._reveal_id = None
            self.results_card.configure(height=0)
            self._set_state_file_selected()

    def process_video_thread(self):
        self._set_state_processing()
        mode = self.mode_var.get()
        threading.Thread(target=self._process_video, args=(mode,), daemon=True).start()

    def _process_video(self, mode_label):
        filters = {
            "player_heatmap": True,
            "rally_length": True,
            "ball_heatmap": True,
            "kitchen_detection": True,
        }

        def update_progress(value):
            self.after(0, lambda: self.progress_bar.set(value))

        try:
            mode_map = {
                "Video Analysis": MODE_VIDEO_ANALYSIS,
                "Split Rallies":  MODE_SPLIT_RALLIES,
                "Detect Serve":   MODE_DETECT_SERVE,
            }
            mode = mode_map[mode_label]
            processor = VideoProcessor(self.video_path, filters, mode=mode)
            processor.process_video(progress_callback=update_progress)

            rally_count   = len(processor.analytics._long_rallies) if hasattr(processor, "analytics") else 0
            serve_results = processor.serve_analyzer.get_results() if hasattr(processor, "serve_analyzer") else []
            serve_count   = len(serve_results)
            serve_avg     = round(sum(r.score for r in serve_results) / serve_count, 1) if serve_count > 0 else 0
            out_dir = processor.output_dir

            self.after(
                0,
                lambda rc=rally_count, sc=serve_count, sa=serve_avg, od=out_dir, m=mode:
                    self._set_state_complete(rc, sc, sa, od, m),
            )
        except Exception as e:
            err = str(e)
            print(f"Error: {err}")
            self.after(0, lambda: self._set_state_error(err))

    def _open_output_folder(self):
        if self.out_dir and os.path.isdir(self.out_dir):
            os.startfile(self.out_dir)


if __name__ == "__main__":
    app = App()
    app.mainloop()
