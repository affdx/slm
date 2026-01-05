export default function AboutPage() {
  return (
    <div className="min-h-screen bg-white dark:bg-slate-950 pt-32 pb-24">
      <div className="container mx-auto px-4">
        <div className="max-w-4xl mx-auto text-center mb-16">
          <h1 className="text-5xl font-bold text-gray-900 dark:text-white mb-6">
            How It Works
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300 leading-relaxed">
            A technical deep-dive into the MSL Translator architecture, inference pipeline, 
            design decisions, and how to extend the system with your own models.
          </p>
        </div>

        {/* Architecture Overview */}
        <section className="mb-20 max-w-5xl mx-auto">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-8 text-center">
            System Architecture
          </h2>
          
          {/* Visual Flowchart */}
          <div className="relative bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 rounded-3xl p-6 px-8 md:p-12 md:px-16 lg:px-20 border border-gray-200 dark:border-gray-700">
            {/* Browser Container Label */}
            <div className="absolute top-4 left-4 flex items-center gap-2">
              <div className="flex gap-1.5">
                <div className="w-3 h-3 rounded-full bg-red-400"></div>
                <div className="w-3 h-3 rounded-full bg-yellow-400"></div>
                <div className="w-3 h-3 rounded-full bg-green-400"></div>
              </div>
              <span className="text-xs font-medium text-gray-500 dark:text-gray-400 ml-2">
                100% Client-side Processing
              </span>
            </div>

            {/* Main Flow */}
            <div className="mt-8 flex flex-col lg:flex-row items-center justify-center gap-4 lg:gap-6 lg:px-8">
              <FlowNode
                icon={
                  <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                  </svg>
                }
                title="Video Input"
                subtitle="Webcam / Upload"
                color="blue"
              />

              <FlowArrow />

              {/* Step 2: MediaPipe */}
              <FlowNode
                icon={
                  <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
                  </svg>
                }
                title="MediaPipe"
                subtitle="258 Landmarks"
                color="purple"
                badge="Holistic"
              />

              <FlowArrow />

              {/* Step 3: Normalize */}
              <FlowNode
                icon={
                  <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01" />
                  </svg>
                }
                title="Normalize"
                subtitle="Z-score / feature"
                color="amber"
              />

              <FlowArrow />

              {/* Step 4: ONNX */}
              <FlowNode
                icon={
                  <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                  </svg>
                }
                title="ONNX Runtime"
                subtitle="BiLSTM Model"
                color="green"
                badge="WASM"
              />

              <FlowArrow />

              {/* Step 5: Output */}
              <FlowNode
                icon={
                  <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                }
                title="Prediction"
                subtitle="90 MSL Glosses"
                color="emerald"
              />
            </div>

            {/* Data Flow Labels */}
            <div className="mt-8 flex flex-wrap justify-center gap-6 text-xs text-gray-500 dark:text-gray-400">
              <DataFlowLabel label="30 FPS" />
              <DataFlowLabel label="[30, 258] tensor" />
              <DataFlowLabel label="~20-50ms inference" />
              <DataFlowLabel label="Top-K + confidence" />
            </div>

            {/* Privacy Badge */}
            <div className="mt-8 flex justify-center">
              <div className="inline-flex items-center gap-2 px-4 py-2 bg-green-100 dark:bg-green-900/30 rounded-full">
                <svg className="w-4 h-4 text-green-600 dark:text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                </svg>
                <span className="text-sm font-medium text-green-700 dark:text-green-300">
                  No data leaves your device
                </span>
              </div>
            </div>
          </div>

          {/* Tech Stack Cards */}
          <div className="grid md:grid-cols-3 gap-6 mt-8">
            <TechCard
              title="MediaPipe Holistic"
              version="0.5.1675471629"
              description="Google's ML framework for real-time pose estimation. Extracts body and hand landmarks from video frames entirely in the browser."
              items={[
                "33 pose landmarks (x, y, z, visibility)",
                "21 left hand landmarks (x, y, z)",
                "21 right hand landmarks (x, y, z)",
                "Total: 258 features per frame",
              ]}
            />
            <TechCard
              title="ONNX Runtime Web"
              version="1.14.0"
              description="Microsoft's cross-platform ML inference engine. Runs PyTorch models in the browser via WebAssembly without needing a server."
              items={[
                "WebAssembly backend (WASM)",
                "Single-threaded for compatibility",
                "SIMD enabled for performance",
                "~4MB model file cached",
              ]}
            />
            <TechCard
              title="BiLSTM Model"
              version="969K params"
              description="Bidirectional LSTM neural network trained on Malaysian Sign Language gestures. Processes 30-frame sequences to classify signs."
              items={[
                "2-layer Bidirectional LSTM",
                "Hidden size: 128",
                "Scalar attention pooling",
                "~93.86% test accuracy",
              ]}
            />
          </div>
        </section>

        {/* Inference Pipeline */}
        <section className="mb-20 max-w-4xl mx-auto">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-8">
            Inference Pipeline
          </h2>

          <div className="space-y-6">
            <PipelineStep
              step={1}
              title="Frame Capture"
              description="Video frames are captured at 30 FPS from webcam or video file. Each frame is drawn to an HTML Canvas element for processing."
            />

            <PipelineStep
              step={2}
              title="Landmark Extraction"
              description="MediaPipe Holistic processes each frame and extracts 258 normalized landmarks (0-1 range) representing body pose and hand positions."
            />

            <PipelineStep
              step={3}
              title="Frame Buffering"
              description="30 consecutive frames are collected into a buffer. If the video is shorter, frames are zero-padded at the beginning to maintain fixed input size."
            />

            <PipelineStep
              step={4}
              title="Per-Feature Normalization"
              description="Each of the 258 features is normalized using pre-computed mean and standard deviation from training data. This ensures consistent input distribution matching what the model was trained on."
            />

            <PipelineStep
              step={5}
              title="ONNX Inference"
              description="The normalized tensor is passed through the BiLSTM model running in WebAssembly. The model outputs logits for 90 classes, which are converted to probabilities via softmax."
            />

            <PipelineStep
              step={6}
              title="Prediction Output"
              description="The class with highest probability is returned along with confidence score and top-K alternatives. A minimum 70% confidence threshold is applied in real-time mode to reduce false positives."
            />
          </div>
        </section>

        {/* Model Architecture */}
        <section className="mb-20 max-w-4xl mx-auto">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-8">
            Model Architecture (BetterLSTM)
          </h2>

          {/* Visual Architecture */}
          <div className="bg-gradient-to-b from-gray-50 to-white dark:from-gray-900 dark:to-gray-800 rounded-2xl p-8 border border-gray-200 dark:border-gray-700 mb-8">
            <div className="flex flex-col items-center gap-4">
              <ArchBlock 
                title="Input" 
                subtitle="[batch, 30, 258]"
                color="gray"
              />
              <ArchArrow />
              <ArchBlock 
                title="Bidirectional LSTM" 
                subtitle="2 layers, hidden=128, dropout=0.35"
                color="blue"
                output="[batch, 30, 256]"
              />
              <ArchArrow />
              <ArchBlock 
                title="LayerNorm" 
                subtitle="Stabilizes gradients"
                color="purple"
                small
              />
              <ArchArrow />
              <ArchBlock 
                title="Scalar Attention Pooling" 
                subtitle="attention = softmax(tanh(Wh) · v)"
                color="amber"
                output="[batch, 256]"
              />
              <ArchArrow />
              <ArchBlock 
                title="MLP Head" 
                subtitle="256 → 256 → 128 → 90 (GELU + Dropout)"
                color="green"
              />
              <ArchArrow />
              <ArchBlock 
                title="Output" 
                subtitle="[batch, 90] logits"
                color="emerald"
              />
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl p-6">
              <h3 className="font-semibold text-lg text-gray-900 dark:text-white mb-4">
                Why Bidirectional?
              </h3>
              <ul className="space-y-2 text-gray-600 dark:text-gray-400 text-sm">
                <li>• Sign meaning depends on full gesture trajectory</li>
                <li>• Forward pass: how gesture starts → develops</li>
                <li>• Backward pass: how gesture ends → originated</li>
                <li>• Combined: richer representation than unidirectional</li>
              </ul>
            </div>

            <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl p-6">
              <h3 className="font-semibold text-lg text-gray-900 dark:text-white mb-4">
                Why GELU Activation?
              </h3>
              <ul className="space-y-2 text-gray-600 dark:text-gray-400 text-sm">
                <li>• Smoother than ReLU (no hard zero cutoff)</li>
                <li>• Used in modern architectures (BERT, GPT)</li>
                <li>• Better gradient flow during training</li>
                <li>• Marginal accuracy improvement over ReLU</li>
              </ul>
            </div>
          </div>
        </section>

        {/* Adding New Models */}
        <section className="mb-20 max-w-4xl mx-auto">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-8">
            Adding a New Model
          </h2>

          <div className="space-y-8">
            <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl p-6">
              <h3 className="font-semibold text-lg text-gray-900 dark:text-white mb-4">
                Step 1: Export to ONNX
              </h3>
              <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 font-mono text-xs mb-4">
                <pre className="text-gray-800 dark:text-gray-200">{`# Export PyTorch model to ONNX format
python scripts/export_onnx.py \\
  --checkpoint models/your_model.pt \\
  --norm-stats models/your_norm_stats.npz \\
  --output web/public/model_yourname.onnx

# This creates:
#   web/public/model_yourname.onnx           - ONNX model
#   web/public/model_yourname_norm_stats.json - Normalization stats`}</pre>
              </div>
              <p className="text-gray-600 dark:text-gray-400 text-sm">
                The export script converts PyTorch to ONNX (opset 14), applies constant folding 
                optimization, and verifies output matches PyTorch within 1e-4 tolerance.
              </p>
            </div>

            <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl p-6">
              <h3 className="font-semibold text-lg text-gray-900 dark:text-white mb-4">
                Step 2: Register in inference.ts
              </h3>
              <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 font-mono text-xs">
                <pre className="text-gray-800 dark:text-gray-200">{`// web/src/lib/inference.ts

export type ModelType = "baseline" | "improved" | "yourmodel";

export const MODEL_CONFIGS: Record<ModelType, ModelConfig> = {
  // ... existing models ...
  yourmodel: {
    name: "Your Model Name",
    description: "Brief description (XX.XX% accuracy)",
    modelPath: "/model_yourname.onnx",
    normStatsPath: "/model_yourname_norm_stats.json",
  },
};`}</pre>
              </div>
            </div>

            <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl p-6">
              <h3 className="font-semibold text-lg text-gray-900 dark:text-white mb-4">
                Step 3: Model Requirements
              </h3>
              <ul className="space-y-3 text-gray-600 dark:text-gray-400 text-sm">
                <li className="flex items-start gap-3">
                  <span className="text-primary-500 font-bold">1.</span>
                  <span><strong>Input shape:</strong> [batch, 30, 258] - must match MediaPipe Holistic output</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-primary-500 font-bold">2.</span>
                  <span><strong>Output shape:</strong> [batch, 90] - raw logits (softmax applied in browser)</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-primary-500 font-bold">3.</span>
                  <span><strong>Normalization stats:</strong> JSON with {`"mean"`} and {`"std"`} arrays (258 floats each)</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-primary-500 font-bold">4.</span>
                  <span><strong>File size:</strong> Keep under 10MB for reasonable load times on mobile</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-primary-500 font-bold">5.</span>
                  <span><strong>ONNX opset:</strong> Version 14 recommended for broad ONNX Runtime compatibility</span>
                </li>
              </ul>
            </div>
          </div>
        </section>

        {/* Running Locally */}
        <section className="mb-20 max-w-4xl mx-auto">
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-8">
            Running Everything Locally
          </h2>

          <div className="grid md:grid-cols-2 gap-6 mb-8">
            <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl p-6">
              <h3 className="font-semibold text-lg text-gray-900 dark:text-white mb-4">
                Web Application
              </h3>
              <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 font-mono text-xs mb-4">
                <pre className="text-gray-800 dark:text-gray-200">{`cd web
npm install
npm run dev

# Open http://localhost:3000`}</pre>
              </div>
              <p className="text-gray-600 dark:text-gray-400 text-sm">
                The web app runs entirely in the browser. Models are loaded from 
                <code className="mx-1 px-1 bg-gray-100 dark:bg-gray-800 rounded">public/</code> 
                on first inference and cached. Works offline after initial load.
              </p>
            </div>

            <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl p-6">
              <h3 className="font-semibold text-lg text-gray-900 dark:text-white mb-4">
                Python API (Optional)
              </h3>
              <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 font-mono text-xs mb-4">
                <pre className="text-gray-800 dark:text-gray-200">{`# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run FastAPI server
uvicorn src.inference.api:app --reload`}</pre>
              </div>
              <p className="text-gray-600 dark:text-gray-400 text-sm">
                Optional server-side API for integration with other services 
                or batch processing of pre-recorded videos.
              </p>
            </div>
          </div>

          <div className="bg-primary-50 dark:bg-primary-900/20 border border-primary-200 dark:border-primary-800 rounded-xl p-6">
            <h3 className="font-semibold text-lg text-gray-900 dark:text-white mb-3">
              Hardware Acceleration
            </h3>
            <div className="grid md:grid-cols-3 gap-4 text-sm">
              <div>
                <h4 className="font-medium text-gray-900 dark:text-white mb-1">Apple Silicon (M1/M2/M3)</h4>
                <p className="text-gray-600 dark:text-gray-400">
                  Training uses MPS backend automatically. Web inference uses WASM (WebGPU coming soon).
                </p>
              </div>
              <div>
                <h4 className="font-medium text-gray-900 dark:text-white mb-1">NVIDIA GPU</h4>
                <p className="text-gray-600 dark:text-gray-400">
                  Training uses CUDA. Web inference uses WASM (WebGL available but WASM more reliable).
                </p>
              </div>
              <div>
                <h4 className="font-medium text-gray-900 dark:text-white mb-1">CPU Only</h4>
                <p className="text-gray-600 dark:text-gray-400">
                  Works everywhere. Training is slower but functional. Inference ~50ms per prediction.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Privacy Note */}
        <section className="max-w-4xl mx-auto">
          <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-xl p-6 text-center">
            <h3 className="font-semibold text-lg text-green-800 dark:text-green-200 mb-2">
              100% Private by Design
            </h3>
            <p className="text-green-700 dark:text-green-300 text-sm max-w-2xl mx-auto">
              All video processing happens in your browser. No frames, landmarks, or predictions 
              are ever sent to any server. Models are downloaded once (~4MB) and cached locally.
              The app works fully offline after initial load.
            </p>
          </div>
        </section>
      </div>
    </div>
  );
}

// Flowchart Components
function FlowNode({ 
  icon, 
  title, 
  subtitle, 
  color,
  badge 
}: { 
  icon: React.ReactNode; 
  title: string; 
  subtitle: string; 
  color: "blue" | "purple" | "amber" | "green" | "emerald";
  badge?: string;
}) {
  const colorStyles = {
    blue: "bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 border-blue-200 dark:border-blue-800",
    purple: "bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400 border-purple-200 dark:border-purple-800",
    amber: "bg-amber-100 dark:bg-amber-900/30 text-amber-600 dark:text-amber-400 border-amber-200 dark:border-amber-800",
    green: "bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400 border-green-200 dark:border-green-800",
    emerald: "bg-emerald-100 dark:bg-emerald-900/30 text-emerald-600 dark:text-emerald-400 border-emerald-200 dark:border-emerald-800",
  };

  const badgeColors = {
    blue: "bg-blue-500 text-white",
    purple: "bg-purple-500 text-white",
    amber: "bg-amber-500 text-white",
    green: "bg-green-500 text-white",
    emerald: "bg-emerald-500 text-white",
  };

  return (
    <div className={`relative flex flex-col items-center p-4 md:p-6 rounded-2xl border-2 ${colorStyles[color]} min-w-[120px] md:min-w-[140px]`}>
      {badge && (
        <span className={`absolute -top-2 -right-2 text-[10px] font-bold px-2 py-0.5 rounded-full ${badgeColors[color]}`}>
          {badge}
        </span>
      )}
      <div className="mb-2">{icon}</div>
      <span className="font-semibold text-sm text-gray-900 dark:text-white text-center">{title}</span>
      <span className="text-xs text-gray-500 dark:text-gray-400 text-center mt-1">{subtitle}</span>
    </div>
  );
}

function FlowArrow() {
  return (
    <div className="flex items-center justify-center text-gray-400 dark:text-gray-500 rotate-90 lg:rotate-0">
      <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
      </svg>
    </div>
  );
}

function DataFlowLabel({ label }: { label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 px-3 py-1 bg-white dark:bg-gray-800 rounded-full border border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-300">
      <span className="w-1.5 h-1.5 bg-primary-500 rounded-full"></span>
      {label}
    </span>
  );
}

// Architecture Block Components
function ArchBlock({ 
  title, 
  subtitle, 
  color, 
  output,
  small 
}: { 
  title: string; 
  subtitle: string; 
  color: "gray" | "blue" | "purple" | "amber" | "green" | "emerald";
  output?: string;
  small?: boolean;
}) {
  const colorStyles = {
    gray: "bg-gray-100 dark:bg-gray-800 border-gray-300 dark:border-gray-600",
    blue: "bg-blue-50 dark:bg-blue-900/20 border-blue-300 dark:border-blue-700",
    purple: "bg-purple-50 dark:bg-purple-900/20 border-purple-300 dark:border-purple-700",
    amber: "bg-amber-50 dark:bg-amber-900/20 border-amber-300 dark:border-amber-700",
    green: "bg-green-50 dark:bg-green-900/20 border-green-300 dark:border-green-700",
    emerald: "bg-emerald-50 dark:bg-emerald-900/20 border-emerald-300 dark:border-emerald-700",
  };

  return (
    <div className={`w-full max-w-md ${small ? 'py-3 px-4' : 'py-4 px-6'} rounded-xl border-2 ${colorStyles[color]} text-center`}>
      <div className="font-semibold text-gray-900 dark:text-white">{title}</div>
      <div className="text-xs text-gray-500 dark:text-gray-400 mt-1 font-mono">{subtitle}</div>
      {output && (
        <div className="text-xs text-primary-600 dark:text-primary-400 mt-2 font-mono">→ {output}</div>
      )}
    </div>
  );
}

function ArchArrow() {
  return (
    <div className="text-gray-400 dark:text-gray-500">
      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
      </svg>
    </div>
  );
}

function TechCard({ title, version, description, items }: { title: string; version: string; description: string; items: string[] }) {
  return (
    <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl p-6">
      <div className="flex items-center justify-between mb-2">
        <h3 className="font-semibold text-gray-900 dark:text-white">{title}</h3>
        <span className="text-xs font-mono bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 px-2 py-1 rounded">
          {version}
        </span>
      </div>
      <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">{description}</p>
      <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
        {items.map((item, i) => (
          <li key={i} className="flex items-start gap-2">
            <span className="text-primary-500 mt-1">•</span>
            <span>{item}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function DecisionCard({ 
  decision, 
  why, 
  alternatives, 
  chosen 
}: { 
  decision: string; 
  why: string; 
  alternatives: { name: string; tradeoff: string }[];
  chosen: string;
}) {
  return (
    <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl p-6">
      <h3 className="font-semibold text-lg text-gray-900 dark:text-white mb-3">
        {decision}
      </h3>
      <p className="text-gray-600 dark:text-gray-400 text-sm mb-4">
        {why}
      </p>
      
      <div className="mb-4">
        <h4 className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-2">
          Alternatives Considered
        </h4>
        <ul className="space-y-2">
          {alternatives.map((alt, i) => (
            <li key={i} className="text-sm">
              <span className="font-medium text-gray-700 dark:text-gray-300">{alt.name}:</span>
              <span className="text-gray-500 dark:text-gray-400 ml-1">{alt.tradeoff}</span>
            </li>
          ))}
        </ul>
      </div>

      <div className="flex items-center gap-2 pt-3 border-t border-gray-100 dark:border-gray-700">
        <span className="text-xs font-semibold text-green-600 dark:text-green-400 uppercase">Chosen:</span>
        <span className="text-sm font-medium text-gray-900 dark:text-white">{chosen}</span>
      </div>
    </div>
  );
}

function PipelineStep({ step, title, description }: { step: number; title: string; description: string }) {
  return (
    <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl p-6">
      <div className="flex items-start gap-4">
        <div className="flex-shrink-0 w-10 h-10 bg-primary-500 text-white rounded-full flex items-center justify-center font-bold text-lg">
          {step}
        </div>
        <div className="flex-1">
          <h3 className="font-semibold text-lg text-gray-900 dark:text-white mb-2">{title}</h3>
          <p className="text-gray-600 dark:text-gray-400 text-sm">{description}</p>
        </div>
      </div>
    </div>
  );
}
