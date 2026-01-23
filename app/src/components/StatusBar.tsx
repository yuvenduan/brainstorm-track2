'use client';

import { cn } from '@/lib/utils';

interface StatusBarProps {
  status: 'disconnected' | 'connecting' | 'connected';
  statusText: string;
  currentTime: number;
  fps: number;
  channelCount: number;
}

export default function StatusBar({ 
  status, 
  statusText, 
  currentTime, 
  fps, 
  channelCount 
}: StatusBarProps) {
  const getStatusColor = () => {
    switch (status) {
      case 'connected':
        return 'bg-success shadow-[0_0_8px_rgba(74,222,128,0.5)]';
      case 'connecting':
        return 'bg-warning shadow-[0_0_8px_rgba(251,191,36,0.5)] animate-pulse-custom';
      default:
        return 'bg-error shadow-[0_0_8px_rgba(248,113,113,0.5)]';
    }
  };

  return (
    <div className="flex items-center gap-3 text-sm text-text-secondary">
      <span className={cn('w-2.5 h-2.5 rounded-full', getStatusColor())}></span>
      <span>{statusText}</span>
      <span className="text-border">|</span>
      <span>t = {currentTime.toFixed(2)}s</span>
      <span className="text-border">|</span>
      <span>{fps} FPS</span>
      <span className="text-border">|</span>
      <span>{channelCount} channels</span>
    </div>
  );
}