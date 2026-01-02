export default function AboutPage() {
  return (
    <div className="min-h-screen bg-white dark:bg-slate-950 pt-32 pb-24">
      <div className="container mx-auto px-4">
        <div className="max-w-4xl mx-auto text-center mb-20">
          <h1 className="text-5xl font-bold text-gray-900 dark:text-white mb-6">
            Bridging Worlds with AI
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300 leading-relaxed">
            MSL Translator is an open-source initiative dedicated to breaking communication barriers 
            for the Malaysian deaf community through accessible technology.
          </p>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-24 max-w-5xl mx-auto">
          <StatCard label="Target Users" value="50k+" sub="Deaf Individuals" />
          <StatCard label="Vocabulary" value="90" sub="Supported Signs" />
          <StatCard label="Latency" value="<200ms" sub="Real-time" />
          <StatCard label="Cost" value="Free" sub="Open Source" />
        </div>

        <div className="grid md:grid-cols-2 gap-12 max-w-5xl mx-auto">
          <section>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">Our Technology</h2>
            <div className="space-y-6">
              <TechItem 
                title="Deep Learning" 
                desc="Bi-Directional LSTM neural network trained on authentic Malaysian Sign Language datasets."
              />
              <TechItem 
                title="Computer Vision" 
                desc="MediaPipe holistic tracking extracts 258 precise landmarks from hands, face, and pose."
              />
              <TechItem 
                title="Edge Computing" 
                desc="Runs entirely in your browser via WebAssembly. No video data is ever sent to a server."
              />
            </div>
          </section>

          <section>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">Impact Areas</h2>
            <div className="grid gap-4">
              <ImpactCard 
                title="Education" 
                icon="🎓"
                desc="Supporting inclusive classrooms and remote learning for deaf students."
              />
              <ImpactCard 
                title="Healthcare" 
                icon="🏥"
                desc="Facilitating critical doctor-patient interactions without barriers."
              />
              <ImpactCard 
                title="Public Services" 
                icon="🏛️"
                desc="Improving accessibility in government offices and public counters."
              />
            </div>
          </section>
        </div>

        <div className="mt-24 max-w-3xl mx-auto text-center">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-8">Community Resources</h2>
          <div className="flex flex-col sm:flex-row justify-center gap-4">
            <a
              href="https://www.mymfd.org.my/"
              target="_blank"
              rel="noopener noreferrer"
              className="px-8 py-4 bg-gray-50 dark:bg-gray-900 rounded-xl hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors font-medium text-gray-900 dark:text-white border border-gray-200 dark:border-gray-800"
            >
              Malaysian Federation of the Deaf
            </a>
            <a
              href="https://www.mybim.com.my/"
              target="_blank"
              rel="noopener noreferrer"
              className="px-8 py-4 bg-gray-50 dark:bg-gray-900 rounded-xl hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors font-medium text-gray-900 dark:text-white border border-gray-200 dark:border-gray-800"
            >
              Learn BIM (MyBIM)
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}

function StatCard({ label, value, sub }: { label: string; value: string; sub: string }) {
  return (
    <div className="p-6 bg-primary-50 dark:bg-primary-900/10 rounded-2xl text-center border border-primary-100 dark:border-primary-800/30">
      <div className="text-3xl font-bold text-primary-600 dark:text-primary-400 mb-1">{value}</div>
      <div className="text-sm font-semibold text-gray-900 dark:text-white">{label}</div>
      <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">{sub}</div>
    </div>
  );
}

function TechItem({ title, desc }: { title: string; desc: string }) {
  return (
    <div className="border-l-2 border-primary-500 pl-4">
      <h3 className="font-semibold text-gray-900 dark:text-white mb-1">{title}</h3>
      <p className="text-gray-600 dark:text-gray-400 text-sm leading-relaxed">{desc}</p>
    </div>
  );
}

function ImpactCard({ title, desc, icon }: { title: string; desc: string; icon: string }) {
  return (
    <div className="flex items-start gap-4 p-4 rounded-xl bg-gray-50 dark:bg-gray-800/50 border border-gray-100 dark:border-gray-800">
      <span className="text-2xl">{icon}</span>
      <div>
        <h3 className="font-semibold text-gray-900 dark:text-white mb-1">{title}</h3>
        <p className="text-sm text-gray-600 dark:text-gray-400">{desc}</p>
      </div>
    </div>
  );
}
