interface AudioPlayerProps {
  audioUrl: string;
  label: string;
}

export default function AudioPlayer({ audioUrl, label }: AudioPlayerProps) {
  return (
    <div className="space-y-2">
      <p className="text-sm font-medium text-gray-300">{label}</p>
      <audio controls className="w-full" src={audioUrl}>
        Your browser does not support the audio element.
      </audio>
    </div>
  );
}
