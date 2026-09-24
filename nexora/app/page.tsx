'use client'

import { Experience } from '@/components/experience/Experience'
import { HUD } from '@/components/hud/HUD'

export default function Home() {
  return (
    <main className="relative min-h-screen">
      {/* 3D Background Canvas */}
      <Experience />

      {/* Persistent UI Overlay */}
      <HUD />

      {/* Scroll timeline container (triggers GSAP) */}
      <div id="scroll-container" className="relative z-0 pointer-events-none">
        {/* We create dummy scroll space for our 15 chapters */}
        {Array.from({ length: 15 }).map((_, i) => (
          <section
            key={i}
            id={`chapter-${(i + 1).toString().padStart(2, '0')}`}
            className="h-[100vh]"
          />
        ))}
      </div>
    </main>
  )
}
