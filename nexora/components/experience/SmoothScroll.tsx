'use client'

import { ReactLenis } from 'lenis/react'
import { ReactNode } from 'react'

export function SmoothScroll({ children }: { children: ReactNode }) {
  return (
    <ReactLenis root options={{ lerp: 0.05, syncTouch: true }}>
      {children}
    </ReactLenis>
  )
}
