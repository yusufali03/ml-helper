# ml-helper

**ml-helper** is a lightweight console assistant for common machine learning tasks, built with a modular plugin architecture, asynchronous execution, and robust logging. It enables you to:

- **Visualize** evaluation metrics (ROC & Precision–Recall curves)
- **Train** a simple neural network on synthetic data
- **Convert** PyTorch models to ONNX format

## 🚀 Features

- **Plugin Architecture (Factory/Registry)**: Easily extendable—add new plugins without modifying core logic.
- **Asynchronous Execution**: Tasks run in isolated child processes using `multiprocessing`, ensuring safe, non-blocking workflows.
- **Error Handling & Logging**: Detailed error classification (parsing, validation, file-not-found, generic) and SQLite-based logging of every run.
- **CLI Usability**: Intuitive commands via `argparse`, helpful `--help`, and exit codes for scripting.
- **Testable End-to-End**: Comprehensive pytest suite covering classification, plugins, and CLI delegation.

## 📁 Project Structure

```plaintext
ml_helper/                 # Project root (GitHub repository)
├── cli.py                 # Delegates task parsing to NLP-based classifier
├── plugin_base.py         # Abstract base class for plugins
├── plugin_factory.py      # Registry & factory to load plugins
├── nlp_classifier.py      # Simple intent detection (ROC, PR, train, convert)
├── main.py                # Entry point; orchestrates parsing, execution, logging
├── logger.py              # SQLite logging helper
├── database.py            # Database initialization
├── plugins/               # Plugin implementations
│   ├── visualization_plugin.py  # ROC & PR curve plots
│   ├── starter_train_plugin.py  # Synthetic-data model training
│   └── model_conversion_plugin.py  # PyTorch → ONNX conversion
├── outputs/               # Saved plots (auto-created)
├── models/                # Saved .pth & .onnx models
└── tests/                 # pytest suite
    └── test_ml_helper.py
```

## 📦 Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/your_username/ml-helper.git
   cd ml-helper
   ```
2. **Create & activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate    # Linux/macOS
   venv\Scripts\activate     # Windows
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

*Note: Ensure **``**, **``**, **``**, and **``** are installed.*

## 🎯 How It Works

1. **CLI Input**: User enters a task, e.g.
   ```bash
   python main.py "build ROC curve from metrics.json"
   ```
2. **Task Parsing** (`nlp_classifier.py`): Detects intent—ROC, Precision–Recall, training, or conversion—extracts parameters.
3. **Plugin Lookup** (`plugin_factory.py`): Maps intent to the corresponding plugin class, which implements `PluginBase.run()`.
4. **Asynchronous Execution** (`multiprocessing` + `Pipe`): Spawns a child process running the plugin to avoid blocking or global state pollution.
5. **Plugin Logic**:
   - **Visualization**: Loads JSON metrics, computes AUC/AP, plots and saves curve.
   - **StarterTrain**: Generates synthetic data, trains a PyTorch model, saves `.pth`.
   - **ModelConversion**: Loads the `.pth`, exports to ONNX.
6. **Error Handling**: Captures exceptions in the child, returns detailed error type and message.
7. **Logging** (`logger.py` + `database.py`): Records each run (timestamp, task, parameters, status, details) into `logs/log.db`.
8. **Result Output**: Prints success or error messages, including output file paths and scores.

## 🛠 Usage Examples

### ROC Curve

```bash
python main.py build ROC curve from metrics.json
```

### Precision–Recall Curve

```bash
python main.py build precision-recall curve from metrics_pr.json
```

### Training Model

```bash
python main.py train model epochs=10 batch_size=16
```

### Convert Model

```bash
python main.py convert models/model.pth to models/model.onnx
```

## ✅ Testing

Run the pytest suite to validate end-to-end functionality:

```bash
pytest -q
```

All core components are covered:

- Task classification
- CLI delegation
- Plugin behaviors
- Error handling

## ✍️ Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m "Add your feature"`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request




