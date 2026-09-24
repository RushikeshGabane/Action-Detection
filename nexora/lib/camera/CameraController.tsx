'use client'

import { useEffect } from 'react'
import { useThree, useFrame } from '@react-three/fiber'
import gsap from 'gsap'
import ScrollTrigger from 'gsap/ScrollTrigger'
import { useWorkflowStore, Chapter } from '../workflow/state'

gsap.registerPlugin(ScrollTrigger)

export function CameraController() {
  const { camera } = useThree()
  const setChapter = useWorkflowStore(state => state.setChapter)

  useEffect(() => {
    // Cinematic chapters based on scroll position
    const chapters: { id: Chapter, trigger: string, pos: [number, number, number], target: [number, number, number] }[] = [
      { id: 'ARRIVAL', trigger: '#chapter-01', pos: [0, 20, 100], target: [0, 0, 60] },
      { id: 'GATE_INTELLIGENCE', trigger: '#chapter-02', pos: [-6, 6, 52], target: [0, 2, 50] },
      { id: 'OCR', trigger: '#chapter-03', pos: [2, 3, 52], target: [0, 1.5, 60] },
      { id: 'INSPECTION', trigger: '#chapter-04', pos: [5, 4, 55], target: [0, 1.5, 60] },
      { id: 'GATE_DECISION', trigger: '#chapter-05', pos: [0, 8, 40], target: [0, 0, 60] },
      { id: 'WEIGHBRIDGE', trigger: '#chapter-06', pos: [8, 10, 25], target: [0, 0, 20] },
      { id: 'YARD_ASSIGNMENT', trigger: '#chapter-07', pos: [0, 50, 0], target: [0, 0, -20] },
      { id: 'YARD_ARRIVAL', trigger: '#chapter-08', pos: [5, 5, -15], target: [0, 1.5, -20] },
      { id: 'PICK_DETECTION', trigger: '#chapter-09', pos: [12, 6, -5], target: [10, 0, -10] },
      { id: 'MOVEMENT_IDENTITY', trigger: '#chapter-10', pos: [10, 8, -5], target: [10, 4, -10] },
      { id: 'LOCATION_UPDATE', trigger: '#chapter-11', pos: [15, 12, -15], target: [0, 0, -20] },
      { id: 'DIGITAL_TWIN', trigger: '#chapter-12', pos: [0, 30, -10], target: [0, 0, -20] },
      { id: 'YARD_MAP', trigger: '#chapter-13', pos: [0, 100, 0], target: [0, 0, 0] },
      { id: 'OPERATIONAL_INTELLIGENCE', trigger: '#chapter-14', pos: [30, 40, 30], target: [0, 0, 0] },
      { id: 'SUMMARY', trigger: '#chapter-15', pos: [0, 60, 80], target: [0, 0, 0] },
    ]

    const tl = gsap.timeline({
      scrollTrigger: {
        trigger: '#scroll-container',
        start: 'top top',
        end: 'bottom bottom',
        scrub: 1,
      }
    })

    chapters.forEach((chapter, index) => {
      tl.to(camera.position, {
        x: chapter.pos[0],
        y: chapter.pos[1],
        z: chapter.pos[2],
        ease: 'power2.inOut',
        onStart: () => setChapter(chapter.id),
        onReverseComplete: () => setChapter(chapter.id)
      }, index)

      // We animate the lookAt by animating a dummy object and updating camera in onUpdate
      // To keep it simple in this mock, we just animate position.
      // In a full app, you would use a proxy object for lookAt.
    })

    return () => {
      ScrollTrigger.getAll().forEach(t => t.kill())
    }
  }, [camera, setChapter])

  useFrame(() => {
    // In a real app we would interpolate lookAt target here based on GSAP proxy
    camera.lookAt(0, 0, 20)
  })

  return null
}
