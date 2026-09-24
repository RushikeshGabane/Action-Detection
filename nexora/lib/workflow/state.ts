import { create } from 'zustand'

export type Chapter =
  | 'ARRIVAL'
  | 'GATE_INTELLIGENCE'
  | 'OCR'
  | 'INSPECTION'
  | 'GATE_DECISION'
  | 'WEIGHBRIDGE'
  | 'YARD_ASSIGNMENT'
  | 'YARD_ARRIVAL'
  | 'PICK_DETECTION'
  | 'MOVEMENT_IDENTITY'
  | 'LOCATION_UPDATE'
  | 'DIGITAL_TWIN'
  | 'YARD_MAP'
  | 'CONTAINER_MAP'
  | 'OPERATIONAL_INTELLIGENCE'
  | 'SUMMARY'

export interface YardLocation {
  block: string;
  row: string;
  cell: string;
  tier: string;
}

export interface ContainerData {
  id: string;
  status: string;
  location: YardLocation;
  lastEvent: string;
  history: string[];
}

interface WorkflowState {
  currentChapter: Chapter;
  activeEvents: string[];
  containerData: ContainerData | null;
  setChapter: (chapter: Chapter) => void;
  addEvent: (event: string) => void;
  clearEvents: () => void;
  setContainerData: (data: ContainerData) => void;
}

export const useWorkflowStore = create<WorkflowState>((set) => ({
  currentChapter: 'ARRIVAL',
  activeEvents: [],
  containerData: {
    id: 'TCLU7423864',
    status: 'EXPECTED',
    location: {
      block: 'C4',
      row: '06',
      cell: '04',
      tier: '02'
    },
    lastEvent: 'NONE',
    history: []
  },
  setChapter: (chapter) => set({ currentChapter: chapter }),
  addEvent: (event) => set((state) => ({ activeEvents: [...state.activeEvents, event] })),
  clearEvents: () => set({ activeEvents: [] }),
  setContainerData: (data) => set({ containerData: data })
}))
