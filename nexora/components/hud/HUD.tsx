'use client'

import { useWorkflowStore, Chapter } from '../../lib/workflow/state'
import { useEffect, useState } from 'react'

export function HUD() {
  const currentChapter = useWorkflowStore(state => state.currentChapter)
  const containerData = useWorkflowStore(state => state.containerData)

  const [mounted, setMounted] = useState(false)
  useEffect(() => { setMounted(true) }, [])

  if (!mounted) return null

  // Helper to map chapter to text
  const getChapterData = (chapter: Chapter) => {
    switch (chapter) {
      case 'ARRIVAL': return { title: 'EVERY MOVE STARTS HERE.', sub: '01 — VEHICLE ARRIVAL' }
      case 'GATE_INTELLIGENCE': return { title: 'SEE BEFORE ENTRY.', sub: '02 — GATE INTELLIGENCE' }
      case 'OCR': return { title: 'READ IN PASSING.', sub: '03 — CONTAINER IDENTIFICATION' }
      case 'INSPECTION': return { title: 'ENTRY HAS EVIDENCE.', sub: '04 — CONDITION INTELLIGENCE' }
      case 'GATE_DECISION': return { title: 'ENTRY BECOMES A DECISION.', sub: '05 — GATE DECISION' }
      case 'WEIGHBRIDGE': return { title: 'EVERY ARRIVAL HAS WEIGHT.', sub: '06 — WEIGHBRIDGE' }
      case 'YARD_ASSIGNMENT': return { title: 'FROM GATE TO CELL.', sub: '07 — YARD ASSIGNMENT' }
      case 'YARD_ARRIVAL': return { title: 'THE CELL IS WAITING.', sub: '08 — YARD ARRIVAL' }
      case 'PICK_DETECTION': return { title: 'PICK BECOMES VISIBLE.', sub: '09 — PICK DETECTION' }
      case 'MOVEMENT_IDENTITY': return { title: 'IDENTITY STAYS WITH THE MOVE.', sub: '10 — MOVEMENT IDENTITY' }
      case 'LOCATION_UPDATE': return { title: 'WHEN IT MOVES, THE RECORD MOVES.', sub: '11 — PLACE + LOCATION UPDATE' }
      case 'DIGITAL_TWIN': return { title: 'ONE CONTAINER. ONE LIVE POSITION.', sub: '12 — CONTAINER INTELLIGENCE' }
      case 'YARD_MAP': return { title: 'NOW SEE THE WHOLE YARD.', sub: '13 — YARD VISIBILITY' }
      case 'OPERATIONAL_INTELLIGENCE': return { title: 'THE OPERATION IS NOW VISIBLE.', sub: '14 — OPERATIONAL INTELLIGENCE' }
      case 'SUMMARY': return { title: 'MAKE EVERY MOVE VISIBLE.', sub: 'NEXORA - AI FOR REAL-WORLD LOGISTICS' }
      default: return { title: '', sub: '' }
    }
  }

  const { title, sub } = getChapterData(currentChapter)

  return (
    <div className="pointer-events-none fixed inset-0 z-10 flex flex-col justify-between p-8 text-white uppercase tracking-wider font-sans">
      {/* Top Header */}
      <header className="flex justify-between items-start text-xs text-gray-400">
        <div>
          <span className="font-bold text-white tracking-widest">NEXORA</span> <br/>
          AI FOR REAL-WORLD LOGISTICS
        </div>
        <div>MENU</div>
      </header>

      {/* Main Chapter Content */}
      <div className="flex-1 flex items-center">
        <div className="max-w-4xl space-y-4">
          <div className="text-[#e63946] text-sm font-medium tracking-[0.2em]">{sub}</div>
          <h1 className="text-5xl md:text-7xl font-bold leading-[0.9] tracking-tighter">
            {title}
          </h1>
        </div>
      </div>

      {/* Intelligence Overlay Data (Glass panel) */}
      <div className="absolute right-8 top-1/2 -translate-y-1/2 w-64 bg-black/40 backdrop-blur-md border border-white/10 p-4 rounded text-xs space-y-4">
        {currentChapter === 'OCR' && (
          <div>
            <div className="text-gray-400 mb-1">OCR TARGET</div>
            <div className="font-bold text-white border-l-2 border-[#e63946] pl-2">CONTAINER NUMBER<br/>{containerData?.id}</div>
            <div className="text-[#e63946] mt-2 animate-pulse">IDENTIFIED</div>
          </div>
        )}
        {currentChapter === 'INSPECTION' && (
          <div>
            <div className="text-gray-400 mb-1">CONDITION CHECK</div>
            <div className="font-bold text-white border-l-2 border-white/20 pl-2">IN PROGRESS</div>
          </div>
        )}
        {currentChapter === 'DIGITAL_TWIN' && containerData && (
          <div>
            <div className="text-gray-400 mb-2">CONTAINER DIGITAL TWIN</div>
            <div className="space-y-1 border-l-2 border-[#e63946] pl-2">
              <div>ID: {containerData.id}</div>
              <div>BLOCK: {containerData.location.block}</div>
              <div>ROW: {containerData.location.row}</div>
              <div>CELL: {containerData.location.cell}</div>
              <div className="text-[#e63946] mt-2">PLACED</div>
            </div>
          </div>
        )}
      </div>

      {/* Footer Progress */}
      <footer className="flex justify-end items-end text-xs text-gray-400">
        <div className="flex items-center gap-4">
          <span>{sub.split('—')[0]?.trim() || '00'} / CURRENT STAGE</span>
          <div className="w-32 h-[1px] bg-white/20 relative">
            {/* Just a dummy progress bar based on chapter rough index */}
            <div className="absolute left-0 top-0 h-full bg-[#e63946] w-1/3"></div>
          </div>
        </div>
      </footer>
    </div>
  )
}
