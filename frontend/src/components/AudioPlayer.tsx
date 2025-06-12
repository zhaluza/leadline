interface AudioPlayerProps {
  audioUrl: string;
  label: string;
}

export default function AudioPlayer({ audioUrl, label }: AudioPlayerProps) {
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <div className="flex-shrink-0 w-6 h-6 bg-green-600 rounded-full flex items-center justify-center">
          <svg
            className="h-3 w-3 text-white"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M14.828 14.828a4 4 0 01-5.656 0M9 10h1m4 0h1m-6 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
        </div>
        <h3 className="text-sm font-semibold text-gray-300">{label}</h3>
      </div>

      <audio controls className="w-full" src={audioUrl} preload="metadata">
        Your browser does not support the audio element.
      </audio>
    </div>
  );
}
