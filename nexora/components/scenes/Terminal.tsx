'use client'

import { useRef } from 'react'
import { Group } from 'three'

export function Terminal() {
  const terminalGroup = useRef<Group>(null)

  // Basic procedural terminal layout:
  // Gate: [0, 0, 50]
  // Weighbridge: [0, 0, 20]
  // Yard: [0, 0, -20]

  return (
    <group ref={terminalGroup}>
      {/* Ground */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <planeGeometry args={[200, 200]} />
        <meshStandardMaterial color="#222222" roughness={0.8} />
      </mesh>

      {/* Grid Helper to give technical feel */}
      <gridHelper args={[200, 40, '#444444', '#222222']} position={[0, 0.1, 0]} />

      {/* Gate Area */}
      <group position={[0, 0, 50]}>
        <mesh position={[-5, 2, 0]} castShadow receiveShadow>
          <boxGeometry args={[1, 4, 1]} />
          <meshStandardMaterial color="#444" />
        </mesh>
        <mesh position={[5, 2, 0]} castShadow receiveShadow>
          <boxGeometry args={[1, 4, 1]} />
          <meshStandardMaterial color="#444" />
        </mesh>
        <mesh position={[0, 4.5, 0]} castShadow receiveShadow>
          <boxGeometry args={[11, 1, 1]} />
          <meshStandardMaterial color="#555" />
        </mesh>
        {/* Gate Camera */}
        <mesh position={[-4, 4, 1]}>
          <boxGeometry args={[0.5, 0.5, 0.5]} />
          <meshStandardMaterial color="#111" />
        </mesh>
      </group>

      {/* Weighbridge */}
      <group position={[0, 0, 20]}>
        <mesh position={[0, 0.2, 0]} receiveShadow>
          <boxGeometry args={[6, 0.4, 12]} />
          <meshStandardMaterial color="#333" />
        </mesh>
        {/* Sensor/Display pole */}
        <mesh position={[-4, 2, 0]} castShadow>
          <boxGeometry args={[0.2, 4, 0.2]} />
          <meshStandardMaterial color="#555" />
        </mesh>
      </group>

      {/* Yard Cells (Blocks) */}
      <group position={[0, 0, -20]}>
        {/* Dummy stacks to simulate a busy yard */}
        {Array.from({ length: 5 }).map((_, i) => (
          Array.from({ length: 5 }).map((_, j) => (
            <group key={`stack-${i}-${j}`} position={[(i - 2) * 10, 1.25, (j - 2) * 15]}>
              <mesh castShadow receiveShadow>
                <boxGeometry args={[3, 2.5, 6]} />
                <meshStandardMaterial color="#444" roughness={0.6} />
              </mesh>
              {Math.random() > 0.5 && (
                <mesh position={[0, 2.6, 0]} castShadow receiveShadow>
                  <boxGeometry args={[3, 2.5, 6]} />
                  <meshStandardMaterial color="#333" roughness={0.6} />
                </mesh>
              )}
            </group>
          ))
        ))}
      </group>

      {/* Active Truck (The one we follow) */}
      <group position={[0, 1.5, 60]}>
        {/* Cabin */}
        <mesh position={[0, 0, -4]} castShadow>
          <boxGeometry args={[2.5, 3, 2.5]} />
          <meshStandardMaterial color="#777" />
        </mesh>
        {/* Trailer/Chassis */}
        <mesh position={[0, -0.5, 0]} castShadow>
          <boxGeometry args={[2.6, 1, 8]} />
          <meshStandardMaterial color="#222" />
        </mesh>
        {/* The Hero Container */}
        <mesh position={[0, 1, 0.5]} castShadow>
          <boxGeometry args={[2.8, 2.6, 6]} />
          <meshStandardMaterial color="#3a4b5c" roughness={0.4} />
        </mesh>
      </group>

      {/* Reach Stacker (Dummy at a yard cell) */}
      <group position={[10, 0, -10]} rotation={[0, -Math.PI / 4, 0]}>
        {/* Base */}
        <mesh position={[0, 2, 0]} castShadow>
          <boxGeometry args={[4, 4, 8]} />
          <meshStandardMaterial color="#e6a519" />
        </mesh>
        {/* Boom */}
        <mesh position={[0, 5, -2]} rotation={[Math.PI / 6, 0, 0]} castShadow>
          <boxGeometry args={[1, 8, 1]} />
          <meshStandardMaterial color="#333" />
        </mesh>
        {/* Spreader */}
        <mesh position={[0, 7.5, 1]} castShadow>
          <boxGeometry args={[3.2, 0.5, 6.2]} />
          <meshStandardMaterial color="#222" />
        </mesh>
      </group>
    </group>
  )
}
