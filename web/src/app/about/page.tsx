export default function AboutPage() {
  return (
    <div className="container mx-auto px-4 py-12">
      <div className="max-w-4xl mx-auto">
        {/* Hero */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
            About MSL Translator
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300">
            Breaking communication barriers for the Malaysian deaf community
          </p>
        </div>

        {/* Mission */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">Our Mission</h2>
          <p className="text-gray-600 dark:text-gray-300 leading-relaxed">
            The MSL Translator project aims to bridge the communication gap between the deaf and
            hearing communities in Malaysia. By leveraging deep learning and computer vision
            technology, we provide an accessible tool that can translate Malaysian Sign Language
            (Bahasa Isyarat Malaysia - BIM) gestures into text, enabling better communication and
            understanding.
          </p>
        </section>

        {/* Technology */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">Technology</h2>
          <div className="grid md:grid-cols-2 gap-6">
            <TechCard
              title="Deep Learning Model"
              description="Our LSTM-based neural network is trained on authentic Malaysian Sign Language data, achieving high accuracy across 90 different sign glosses."
            />
            <TechCard
              title="MediaPipe Landmarks"
              description="We use Google's MediaPipe framework to extract 258 hand and body landmarks from video frames, providing robust gesture recognition."
            />
            <TechCard
              title="Real-time Processing"
              description="Optimized for speed with inference times under 200ms, enabling real-time translation for practical everyday use."
            />
            <TechCard
              title="Accessible Design"
              description="Built with accessibility in mind, featuring high contrast modes, keyboard navigation, and screen reader support."
            />
          </div>
        </section>

        {/* Impact */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">Social Impact</h2>
          <div className="bg-primary-50 dark:bg-primary-900/20 rounded-xl p-6">
            <div className="grid md:grid-cols-3 gap-6 text-center">
              <div>
                <p className="text-4xl font-bold text-primary-600 dark:text-primary-400">~50,000</p>
                <p className="text-gray-600 dark:text-gray-300 mt-2">
                  Deaf individuals in Malaysia who could benefit
                </p>
              </div>
              <div>
                <p className="text-4xl font-bold text-primary-600 dark:text-primary-400">90</p>
                <p className="text-gray-600 dark:text-gray-300 mt-2">
                  Sign language glosses supported
                </p>
              </div>
              <div>
                <p className="text-4xl font-bold text-primary-600 dark:text-primary-400">Free</p>
                <p className="text-gray-600 dark:text-gray-300 mt-2">
                  Open-source and accessible to all
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Use Cases */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">Use Cases</h2>
          <ul className="space-y-4">
            <UseCaseItem
              title="Education"
              description="Help teachers and students communicate more effectively in inclusive classrooms."
            />
            <UseCaseItem
              title="Healthcare"
              description="Enable better patient-doctor communication in medical settings."
            />
            <UseCaseItem
              title="Public Services"
              description="Improve accessibility at government offices, banks, and other public services."
            />
            <UseCaseItem
              title="Daily Life"
              description="Facilitate everyday conversations between deaf and hearing individuals."
            />
          </ul>
        </section>

        {/* Resources */}
        <section className="mb-12">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">Resources</h2>
          <div className="grid md:grid-cols-2 gap-4">
            <a
              href="https://www.mymfd.org.my/"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center p-4 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 hover:border-primary-500 transition-colors"
            >
              <div>
                <p className="font-medium text-gray-900 dark:text-white">
                  Malaysian Federation of the Deaf
                </p>
                <p className="text-sm text-gray-500 dark:text-gray-400">mymfd.org.my</p>
              </div>
            </a>
            <a
              href="https://www.mybim.com.my/"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center p-4 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 hover:border-primary-500 transition-colors"
            >
              <div>
                <p className="font-medium text-gray-900 dark:text-white">
                  Bahasa Isyarat Malaysia
                </p>
                <p className="text-sm text-gray-500 dark:text-gray-400">mybim.com.my</p>
              </div>
            </a>
          </div>
        </section>

        {/* Credits */}
        <section className="border-t border-gray-200 dark:border-gray-700 pt-8">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">Acknowledgments</h2>
          <p className="text-gray-600 dark:text-gray-300 leading-relaxed">
            This project was developed as part of an academic research initiative to advance
            accessibility technology in Malaysia. We thank the Malaysian deaf community for their
            support and the researchers who contributed to the sign language dataset.
          </p>
        </section>
      </div>
    </div>
  );
}

function TechCard({ title, description }: { title: string; description: string }) {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
      <h3 className="font-semibold text-gray-900 dark:text-white mb-2">{title}</h3>
      <p className="text-gray-600 dark:text-gray-300 text-sm">{description}</p>
    </div>
  );
}

function UseCaseItem({ title, description }: { title: string; description: string }) {
  return (
    <li className="flex items-start space-x-3">
      <svg
        className="w-6 h-6 text-primary-600 dark:text-primary-400 flex-shrink-0 mt-0.5"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
        />
      </svg>
      <div>
        <p className="font-medium text-gray-900 dark:text-white">{title}</p>
        <p className="text-gray-600 dark:text-gray-300 text-sm">{description}</p>
      </div>
    </li>
  );
}
