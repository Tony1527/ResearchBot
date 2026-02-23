"""PyQt-based front end that wraps the ResearchBot agent graph."""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QThread, pyqtSignal, Qt, QEvent
from PyQt5.QtGui import QColor, QTextCursor, QFont
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QLineEdit,
    QPushButton,
    QLabel,
    QComboBox,
    QGroupBox,
    QFormLayout,
    QCheckBox,
    QSpinBox,
    QFileDialog,
    QMessageBox,
    QSplitter,
    QFrame,
    QScrollArea,
)

from ResearchBot import ResearchBot as RB, ResearchBotConfig
from utility import stream_response


CONFIG_FILE = Path(__file__).parent / "gui_config.json"

DEFAULT_MODELS = [
    "deepseek/deepseek-chat",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "google/gemini-pro",
]


class StreamWorker(QThread):
    """Runs the async streaming call without blocking the Qt event loop."""
    finished = pyqtSignal(str)
    token_received = pyqtSignal(str)
    # thinking_updated = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, bot, user_input):
        super().__init__()
        self.bot = bot
        self.user_input = user_input

    async def run_async(self):
        """Invoke the graph coroutine and propagate completion/errors."""
        try:
            await self.bot.stream_response(self.user_input, callback=self.token_received.emit)
            self.finished.emit("Done")
        except Exception as e:
            self.error_occurred.emit(str(e))

    def run(self):
        """Bridge Qt thread lifecycle into an isolated asyncio loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.run_async())
        finally:
            loop.close()


class AsyncWorker(QThread):
    """Runs the async document call without blocking the Qt event loop."""
    finished = pyqtSignal(str)
    signal = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    func = None
    args = None

    def __init__(self, func, *args):
        super().__init__()
        self.func = func
        self.args = args

    async def run_async(self):
        """Invoke the graph coroutine and propagate completion/errors."""
        try:
            await self.func(*self.args)
            self.finished.emit("Done")
        except Exception as e:
            self.error_occurred.emit(str(e))

    def run(self):
        """Bridge Qt thread lifecycle into an isolated asyncio loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.run_async())
        finally:
            loop.close()


class ResearchBotGUI(QMainWindow):
    """High-level widget that wires inputs, settings, and bot streaming."""
    def __init__(self):
        super().__init__()
        self.bot = None
        # self.is_thinking = False
        self.stream_worker = None
        self.document_worker = None
        self.init_ui()
        self.load_config()
        self.initialize_bot()

    def init_ui(self):
        """Build the splitter layout hosting chat view and settings panel."""

        ## create main layout with horizontal splitter
        self.setWindowTitle("ResearchBot")
        self.setGeometry(100, 100, 900, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        ## create chat widget and settings widget, then add them to the splitter
        self.chat_widget = self.create_chat_widget()
        self.settings_widget = self.create_settings_widget()

        splitter.addWidget(self.chat_widget)
        splitter.addWidget(self.settings_widget)
        ## set the sizes for chat and settings panels (3:1 ratio)
        splitter.setSizes([600, 300])

        ## adjust stretch factors to maintain the ratio when resizing
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

    def create_chat_widget(self) -> QWidget:
        """Construct the chat area containing transcript and message input."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)

        # title_label = QLabel("ResearchBot Chat")
        # title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        # layout.addWidget(title_label)

        # self.thinking_label = QLabel("Thinking: ")
        # self.thinking_label.setStyleSheet("color: #888800; font-style: italic;")
        # self.thinking_label.setWordWrap(True)
        # self.thinking_label.hide()
        # layout.addWidget(self.thinking_label)

        # thinking_group = QGroupBox("Thinking Process")
        # thinking_layout = QVBoxLayout()
        # self.thinking_text = QTextEdit()
        # self.thinking_text.setReadOnly(True)
        # self.thinking_text.setStyleSheet("""
        #     QTextEdit {
        #         background-color: #1e1e1e;
        #         color: #ffcc00;
        #         font-family: 'Courier New', monospace;
        #         font-size: 12px;
        #     }
        # """)
        # thinking_layout.addWidget(self.thinking_text)
        # thinking_group.setLayout(thinking_layout)
        # thinking_group.hide()
        # layout.addWidget(thinking_group)
        # self.thinking_group = thinking_group

        chat_group = QGroupBox("")
        chat_layout = QVBoxLayout()

        ## display area for the chat conversation
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setMinimumHeight(400)
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background-color: #f5f5f5;
                color: #000000;
                font-size: 14px;
            }
        """)
        chat_layout.addWidget(self.chat_display)



        ## input field
        input_layout = QHBoxLayout()
        self.input_field = QTextEdit()
        self.input_field.setMinimumHeight(10)  # 设置输入框高度
        self.input_field.setPlaceholderText("Enter your message...")
        self.input_field.textChanged.connect(self.handle_input_changed)
        self.input_field.installEventFilter(self)
        # [建议] 对输入框也做同样的修改，以防类似问题
        self.input_field.setStyleSheet("""
            QTextEdit {
                background-color: #ffffff;
                color: #000000;
            }
        """)

        input_layout.addWidget(self.input_field, 4)

        chat_layout.addLayout(input_layout)
        chat_group.setLayout(chat_layout)
        layout.addWidget(chat_group)

        return widget

    def create_settings_widget(self) -> QWidget:
        """Assemble controls for model selection, API settings, and docs."""
        widget = QWidget()
        widget.setMaximumWidth(400)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)

        title_label = QLabel("Settings")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title_label)



        model_group = QGroupBox("")
        model_layout = QFormLayout()

        self.model_combo = QComboBox()
        self.model_combo.addItems(DEFAULT_MODELS)
        self.model_combo.setEditable(True)
        self.model_combo.setCurrentText("deepseek/deepseek-chat")
        model_layout.addRow("LLM Model:", self.model_combo)

        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setPlaceholderText("Enter API Key")
        model_layout.addRow("API Key:", self.api_key_input)

        # self.api_base_input = QLineEdit()
        # self.api_base_input.setPlaceholderText("(optional)")
        # model_layout.addRow("API Base:", self.api_base_input)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)



        email_group = QGroupBox("")
        email_layout = QFormLayout()

        self.email_addr = QLineEdit()
        self.email_addr.setPlaceholderText("Email Address (Optional)")
        email_layout.addRow("Email Agent Address:", self.email_addr)

        self.email_api_key_input = QLineEdit()
        self.email_api_key_input.setEchoMode(QLineEdit.Password)
        self.email_api_key_input.setPlaceholderText("API Key (Optional)")
        email_layout.addRow("Email Agent:", self.email_api_key_input)

        email_group.setLayout(email_layout)
        layout.addWidget(email_group)





        collection_group = QGroupBox("Local documents")
        collection_layout = QFormLayout()

        add_doc_btn = QPushButton("Add Files")
        add_doc_btn.clicked.connect(self.add_documents)
        
        add_dir_btn = QPushButton("Add Dir")
        add_dir_btn.clicked.connect(self.add_directory)

        collection_layout.addWidget(add_doc_btn)
        collection_layout.addWidget(add_dir_btn)

        collection_group.setLayout(collection_layout)
        layout.addWidget(collection_group)

        # self.init_bot_button = QPushButton("Initialize Bot")
        # self.init_bot_button.clicked.connect(self.initialize_bot)
        # layout.addWidget(self.init_bot_button)

        # self.status_label = QLabel("Status: Not initialized")
        # self.status_label.setStyleSheet("color: red;")
        # layout.addWidget(self.status_label)

        layout.addStretch()

        

        # save_config_btn = QPushButton("Save Configuration")
        # save_config_btn.clicked.connect(self.save_config)
        # layout.addWidget(save_config_btn)

        return widget

    # def toggle_embedding_model(self, state):
    #     self.embedding_model_input.setEnabled(
    #         not self.local_embedding_check.isChecked()
    #     )

    # def toggle_reranker_model(self, state):
    #     self.reranker_model_input.setEnabled(not self.local_reranker_check.isChecked())

    def initialize_bot(self):
        """Instantiate ResearchBot with current UI values and guard inputs."""
        try:
            llm_api_key = self.api_key_input.text().strip()
            email_api_key = self.email_api_key_input.text().strip()

            # api_base = self.api_base_input.text().strip()
            if not llm_api_key:
                QMessageBox.warning(self, "Warning", "Please enter an API key!")
                return None

            os.environ["AGENTMAIL_API_KEY"] = email_api_key

            # if api_base:
            #     os.environ["LITELLM_BASE_URL"] = api_base

            

            config = ResearchBotConfig(
                llm_model=self.model_combo.currentText(),
                llm_api_key=llm_api_key,
                # llm_api_base=api_base,
                # embedding_model=self.embedding_model_input.text(),
                # cross_encoder_model=self.reranker_model_input.text(),
                # is_local_embedding=self.local_embedding_check.isChecked(),
                # is_local_cross_encoder=self.local_reranker_check.isChecked(),
                # max_sources=self.max_sources_spin.value(),
                # chunk_chars=self.chunk_chars_spin.value(),
                # overlap=self.overlap_spin.value(),
                # persist_directory=self.persist_dir_input.text(),
                streaming=True,
                email_addr=self.email_addr.text().strip() or None,
            )

            self.bot = RB(config)
            # self.status_label.setText("Status: Initialized")
            # self.status_label.setStyleSheet("color: green;")
            # self.init_bot_button.setEnabled(False)
            # self.init_bot_button.setText("Bot Initialized")
            return self.bot

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to initialize bot: {str(e)}")
            return None

    def add_documents(self):
        """Let the user import PDFs and push them through the bot's ingestion."""
        if not self.bot:
            QMessageBox.warning(self, "Warning", "Please initialize the bot first!")
            return


        files, _ = QFileDialog.getOpenFileNames(
            self, "Select PDF files", "", "PDF Files (*.pdf)"
        )

        self.add_files2bot(files)

    def add_files2bot(self, files):
        if files:
            self.append_message("System", f"Adding documents...\n")

            self.document_worker = AsyncWorker(self.bot.aadd, files)
            self.document_worker.finished.connect(lambda msg: self.on_finished(sender="System",msg=f"Finish adding files: {files}\n"))
            self.document_worker.error_occurred.connect(self.on_error)
            self.document_worker.start()

    def add_directory(self):
        """Let the user select a directory and ingest all PDFs within it."""
        if not self.bot:
            QMessageBox.warning(self, "Warning", "Please initialize the bot first!")
            return

        directory = QFileDialog.getExistingDirectory(self, "Select Directory")

        self.add_files2bot(directory)

    def handle_input_changed(self):
        pass

    def eventFilter(self, obj, event):
        """Send message on Enter while allowing modifiers to insert new lines."""
        if obj == self.input_field and event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Return:
                if (
                    event.modifiers() & (Qt.ControlModifier|Qt.MetaModifier|Qt.ShiftModifier)
                ):
                    return False  # Command+Return 或 Ctrl+Return = 换行
                else:
                    if self.bot is None:
                        # print("Correct position!")
                        # self.append_message("System", "Bot is waking up...\n\n")
                        
                        # self.chat_display.repaint()
                        # QApplication.processEvents()
                        bot = self.initialize_bot()
                        if bot:
                            self.send_message()
                        else:
                            self.append_message("System", "Bot continues to sleep.\n\n")
                    else:
                        # print("Wrong position!")
                        self.send_message()
                    return True
        return super().eventFilter(obj, event)
    
    # def deferred_init_and_send(self):
    #     """Helper to initialize and then trigger the message sending."""
    #     # 这里执行耗时的同步初始化
    #     # 注意：为了彻底不卡顿，initialize_bot 最好也放到线程里，
    #     # 但既然它是同步的，QTimer 至少能保证 'Initializing...' 消息先显示出来。
    #     QApplication.setOverrideCursor(Qt.WaitCursor) # 显示忙碌光标
    #     try:
    #         self.initialize_bot()
    #         if self.bot:
    #             self.send_message()
    #     finally:
    #         QApplication.restoreOverrideCursor()

    def append_bot_message(self, message: str):
        # 1. 获取垂直滚动条
        scrollbar = self.chat_display.verticalScrollBar()
        
        # 2. 判断当前是否正好在底部（或者非常接近底部）
        # scrollbar.value() 是当前位置，scrollbar.maximum() 是最大可滚动值
        was_at_bottom = (scrollbar.value() >= (scrollbar.maximum() - 10))
        
        # 3. 移动光标并插入文本（这是标准操作）
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(message)
        
        # 4. 根据之前的状态决定是否滚动
        if was_at_bottom:
            # 如果之前就在底部，说明用户正在看最新内容，那么保持跟随
            self.chat_display.setTextCursor(cursor) # 将光标更新为末尾的光标
            self.chat_display.ensureCursorVisible()
        else:
            # 如果之前不在底部（用户可能滚上去看旧消息了），
            # 就不调用 ensureCursorVisible()，也不更新显示的 cursor 位置，
            # 这样界面就不会跳动。
            pass


        # cursor = self.chat_display.textCursor()
        # cursor.movePosition(QTextCursor.End)
        # cursor.insertText(message)
        # self.chat_display.ensureCursorVisible()

    def send_message(self):
        """Kick off a worker thread to stream bot replies for current prompt."""
        if not self.bot:
            QMessageBox.warning(self, "Warning", "Please initialize the bot first!")
            return

        user_input = self.input_field.toPlainText()+"\n"
        if not user_input:
            return

        self.input_field.clear()
        self.append_message("You", user_input)

        ## add a placeholder bot message to create the "Bot:" line in the chat before streaming starts
        self.append_message("Bot", "")

        self.stream_worker = StreamWorker(self.bot, user_input)
        self.stream_worker.token_received.connect(self.append_bot_message)
        self.stream_worker.finished.connect(self.on_finished)
        self.stream_worker.error_occurred.connect(self.on_error)
        self.bot.set_callback(self.stream_worker.token_received.emit)
        self.stream_worker.start()


    def on_finished(self, result=None, msg="", sender="_"):
        """Notify the UI once the asynchronous exchange completes."""
        pass
        # self.is_thinking = False
        # self.send_button.setEnabled(True)
        # self.thinking_group.hide()
        # self.thinking_label.hide()
        self.append_message(sender, msg)
        self.append_message("_", "========================================\n\n")

    def on_error(self, error: str):
        """Bubble worker exceptions into both chat transcript and dialog."""
        # self.is_thinking = False
        # self.send_button.setEnabled(True)
        # self.thinking_group.hide()
        # self.thinking_label.hide()
        self.append_message("Error", error)
        QMessageBox.critical(self, "Error", f"Error occurred: {error}")
        self.append_message("_", "========================================\n\n")

    def append_message(self, sender: str, message: str):
        """Color-code chat lines so users can distinguish speaker roles."""
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)

        # Preserve line breaks in HTML
        formatted_message = message.replace("\n", "<br>")  

        if sender == "You":
            color = "#0066cc"
        elif sender == "Bot":
            color = "#008800"
        elif sender == "System":
            color = "#000000"
        elif sender == "Error":
            color = "#cc0000"
        elif sender == "_":
            color = "#FFFFFF"
        else:
            color = "#666666"

        if sender == "_":
            formatted_message = f'<p style="margin: 5px 0;">{formatted_message}</p>'
        else:
            formatted_message = f'<p style="margin: 5px 0;"><b style="color: {color};">{sender}:</b> {formatted_message}</p>'
        cursor.insertHtml(formatted_message)
        
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()


    def load_config(self):
        """Restore persisted GUI preferences if the json config exists."""
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r") as f:
                    config = json.load(f)

                self.model_combo.setCurrentText(
                    config.get("llm_model", "deepseek/deepseek-chat")
                )
                self.api_key_input.setText(config.get("llm_api_key", ""))
                self.email_api_key_input.setText(config.get("email_api_key", ""))
                self.email_addr.setText(config.get("email_addr", ""))
                # self.api_base_input.setText(config.get("llm_api_base", ""))
                # self.embedding_model_input.setText(
                #     config.get("embedding_model", "BAAI/bge-m3")
                # )
                # self.reranker_model_input.setText(
                #     config.get("cross_encoder_model", "BAAI/bge-reranker-base")
                # )
                # self.local_embedding_check.setChecked(
                #     config.get("is_local_embedding", True)
                # )
                # self.local_reranker_check.setChecked(
                #     config.get("is_local_cross_encoder", True)
                # )
                # self.max_sources_spin.setValue(config.get("max_sources", 10))
                # self.chunk_chars_spin.setValue(config.get("chunk_chars", 500))
                # self.overlap_spin.setValue(config.get("overlap", 100))
                # self.persist_dir_input.setText(
                #     config.get("persist_directory", "./output/collection")
                # )
            except Exception as e:
                print(f"Failed to load config: {e}")

    def save_config(self):
        """Persist current settings so the next session can reuse them."""
        config = {
            "llm_model": self.model_combo.currentText(),
            "llm_api_key": self.api_key_input.text().strip(),
            "email_api_key": self.email_api_key_input.text().strip(),
            "email_addr": self.email_addr.text().strip() or None
        }

        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=2)
            # QMessageBox.information(self, "Success", "Configuration saved!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save config: {str(e)}")

    def closeEvent(self, event):
        """Stop any running worker thread and dump config before exit."""
        if self.stream_worker and self.stream_worker.isRunning():
            self.stream_worker.terminate()
            self.stream_worker.wait()

        if self.document_worker and self.document_worker.isRunning():
            self.document_worker.terminate()
            self.document_worker.wait()

        self.save_config()
        event.accept()


def main():
    """Bootstrap the Qt application and launch the main window."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    palette = app.palette()
    palette.setColor(palette.Window, QColor(240, 240, 240))
    app.setPalette(palette)

    window = ResearchBotGUI()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
