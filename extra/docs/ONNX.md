# What is ONNX and why does it exist?

Imagine you train a model in Python with XGBoost. Your client’s production system runs in Java. Or C++. Or their mobile app runs on-device inference. Your `model.pkl` is useless to them — it’s Python-specific, XGBoost-specific, and requires the entire sklearn/XGBoost stack to load.

**ONNX (Open Neural Network Exchange)** is a universal model format. It’s an open standard that describes a model’s computation graph in a way that’s language and framework agnostic. Once you export to ONNX, any runtime that speaks ONNX can run your model — Java, C#, C++, mobile, edge devices, browsers.

In practice for you as an API engineer, the main benefit is **inference speed**. ONNX Runtime (Microsoft’s engine) is heavily optimized for fast CPU inference — typically 2-5x faster than calling model.predict() through the Python XGBoost API. For a high-traffic API serving thousands of requests, that matters.


## The ONNX mental model

Your XGBoost model internally is a collection of decision trees — each tree is a series of if/else splits. ONNX represents those splits as a computation graph of standardized operators. `onnxruntime` then executes that graph with optimized C++ code instead of Python.

```text
XGBoost model (.pkl)          ONNX model (.onnx)
Python-dependent         →    Language agnostic
XGBoost runtime needed   →    ONNX Runtime (any language)
~slower inference        →    ~2-5x faster inference
```

What you need to install
```bash
pip install skl2onnx onnxruntime onnx
```

skl2onnx converts sklearn-compatible models (including XGBoost) to ONNX format. onnxruntime runs them. Add these to your requirements.txt too.