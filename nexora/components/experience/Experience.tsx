'use client'

import { Canvas } from '@react-three/fiber'
import { Terminal } from '../scenes/Terminal'
import { CameraController } from '../../lib/camera/CameraController'

export function Experience() {
  return (
    <div className="fixed inset-0 z-0 bg-[#111111]">
      <Canvas
        shadows
        gl={{ antialias: true, powerPreference: 'high-performance' }}
        camera={{ position: [0, 50, 100], fov: 45 }}
      >
        <color attach="background" args={['#111111']} />
        <fog attach="fog" args={['#111111', 50, 200]} />

        <ambientLight intensity={0.5} />
        <directionalLight
          position={[100, 100, 50]}
          intensity={1.5}
          castShadow
          shadow-mapSize={[2048, 2048]}
        />

        <Terminal />
        <CameraController />
      </Canvas>
    </div>
  )
}
