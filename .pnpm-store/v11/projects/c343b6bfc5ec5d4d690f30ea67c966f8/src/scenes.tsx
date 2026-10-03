import React, { Suspense, useEffect, useMemo, useRef, useState } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'

// ─── Interactive volatility field ─────────────────────────────────────
// Drag to orbit · hover a candle to inspect its range · click for a burst.
// Stays on-brand (teal = calm range, ember = volatile range) and keeps the
// "illustrative geometry, not live prices" honesty of the previous scene.
type CandleDatum = {
  x: number; open: number; close: number; high: number; low: number
  mid: number; body: number; vol: number; up: boolean; regime: 'CALM' | 'WIDE'
}

const FIELD_N = 54
const TEAL = '#3E7C74'
const TEAL_LT = '#78A99F'
const EMBER = '#E5462B'
const EMBER_LT = '#F05A40'

function buildField(): CandleDatum[] {
  return Array.from({ length: FIELD_N }, (_, i) => {
    const drift = Math.sin(i * 0.32) * 0.34 + Math.sin(i * 0.11) * 0.52 + i * 0.016
    const regimeMix = Math.sin(i * 0.23 + 1.2) * 0.5 + 0.5 // 0..1 slow regime cycle
    const open = drift + Math.sin(i * 2.1) * 0.15
    const close = drift + Math.cos(i * 1.65) * 0.16
    const body = Math.max(0.07, Math.abs(close - open))
    const vol = 0.16 + regimeMix * 0.42 + ((i * 37) % 7) * 0.022
    const high = Math.max(open, close) + vol * 0.5
    const low = Math.min(open, close) - vol * 0.5
    return {
      x: (i - (FIELD_N - 1) / 2) * 0.104,
      open, close, high, low,
      mid: (open + close) / 2,
      body, vol,
      up: close >= open,
      regime: vol > 0.36 ? 'WIDE' : 'CALM',
    }
  })
}

export type HoverInfo = { index: number; datum: CandleDatum } | null

function VolatilityField({ reducedMotion = false, hovered, onHover, onBurst }: {
  reducedMotion?: boolean; hovered: HoverInfo; onHover: (h: HoverInfo) => void; onBurst: () => void
}) {
  const candles = useMemo(buildField, [])
  const group = useRef<THREE.Group>(null!)
  const tilt = useRef<THREE.Group>(null!)
  const bodies = useRef<THREE.InstancedMesh>(null!)
  const wicks = useRef<THREE.InstancedMesh>(null!)
  const diamonds = useRef<THREE.InstancedMesh>(null!)
  const mouseLight = useRef<THREE.PointLight>(null!)
  const hoverGlow = useRef<THREE.Mesh>(null!)
  const ringA = useRef<THREE.Mesh>(null!)
  const ringB = useRef<THREE.Mesh>(null!)
  const pulses = useRef<Array<THREE.Mesh | null>>([])
  const dust = useRef<THREE.Points>(null!)
  const { gl } = useThree()

  // Drag-to-orbit state (mutable refs so we never re-render per mousemove).
  const drag = useRef({ active: false, moved: 0, lastX: 0, lastY: 0, ry: 0, rx: 0, vy: 0, vx: 0 })
  const burst = useRef<{ index: number; start: number } | null>(null)
  const clockT = useRef(0)
  const elapsed = useRef(0)
  const hoveredIdx = useRef<number | null>(null)
  hoveredIdx.current = hovered?.index ?? null

  const wideIdx = useMemo(() => candles.map((c, i) => (c.regime === 'WIDE' ? i : -1)).filter(i => i >= 0), [candles])

  const trendCurve = useMemo(() => {
    const pts = candles.map(c => new THREE.Vector3(c.x, Math.max(c.open, c.close) + 0.2, 0.06))
    return new THREE.CatmullRomCurve3(pts)
  }, [candles])
  const trendTube = useMemo(() => new THREE.TubeGeometry(trendCurve, 96, 0.014, 8, false), [trendCurve])
  useEffect(() => () => trendTube.dispose(), [trendTube])

  const ribbon = useMemo(() => {
    // Translucent band between the high and low envelopes = the "range regime".
    const pos: number[] = []
    const idx: number[] = []
    candles.forEach((c, i) => {
      pos.push(c.x, c.high + 0.1, -0.06, c.x, c.low - 0.1, -0.06)
      if (i < candles.length - 1) {
        const a = i * 2
        idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2)
      }
    })
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pos), 3))
    g.setIndex(idx)
    g.computeVertexNormals()
    return g
  }, [candles])
  useEffect(() => () => ribbon.dispose(), [ribbon])

  const dustPos = useMemo(() => {
    const arr = new Float32Array(220 * 3)
    for (let i = 0; i < 220; i++) {
      arr[i * 3] = (Math.random() - 0.5) * 7
      arr[i * 3 + 1] = (Math.random() - 0.5) * 4.4
      arr[i * 3 + 2] = -0.4 - Math.random() * 2.2
    }
    return arr
  }, [])

  const grid = useMemo(() => {
    const points: THREE.Vector3[] = []
    for (let y = -1.35; y <= 1.6; y += 0.5) {
      points.push(new THREE.Vector3(-3.1, y, -0.22), new THREE.Vector3(3.1, y, -0.22))
    }
    for (let x = -3; x <= 3; x += 0.5) {
      points.push(new THREE.Vector3(x, -1.45, -0.22), new THREE.Vector3(x, 1.8, -0.22))
    }
    return points
  }, [])

  // Base instance colors (set once).
  useEffect(() => {
    if (!bodies.current || !wicks.current || !diamonds.current) return
    const m = new THREE.Object3D()
    const col = new THREE.Color()
    candles.forEach((c, i) => {
      m.position.set(c.x, c.mid, 0)
      m.scale.set(1, c.body, 1)
      m.updateMatrix()
      bodies.current.setMatrixAt(i, m.matrix)
      col.set(c.regime === 'WIDE' ? EMBER : TEAL).lerp(new THREE.Color(c.up ? TEAL_LT : EMBER_LT), 0.35)
      bodies.current.setColorAt(i, col)
      m.position.set(c.x, (c.high + c.low) / 2, -0.012)
      m.scale.set(0.24, c.high - c.low, 1)
      m.updateMatrix()
      wicks.current.setMatrixAt(i, m.matrix)
      wicks.current.setColorAt(i, col)
    })
    wideIdx.forEach((ci, k) => {
      const c = candles[ci]
      m.position.set(c.x, c.high + 0.32, 0.02)
      m.scale.setScalar(1)
      m.rotation.set(0, 0, 0)
      m.updateMatrix()
      diamonds.current.setMatrixAt(k, m.matrix)
      diamonds.current.setColorAt(k, new THREE.Color(EMBER_LT))
    })
    bodies.current.instanceMatrix.needsUpdate = true
    wicks.current.instanceMatrix.needsUpdate = true
    diamonds.current.instanceMatrix.needsUpdate = true
    if (bodies.current.instanceColor) bodies.current.instanceColor.needsUpdate = true
    if (wicks.current.instanceColor) wicks.current.instanceColor.needsUpdate = true
    if (diamonds.current.instanceColor) diamonds.current.instanceColor.needsUpdate = true
  }, [candles, wideIdx])

  // Drag listeners on the canvas element.
  useEffect(() => {
    const el = gl.domElement
    const d = drag.current
    const down = (e: PointerEvent) => {
      d.active = true; d.moved = 0; d.lastX = e.clientX; d.lastY = e.clientY; d.vy = 0; d.vx = 0
      el.style.cursor = 'grabbing'
      el.setPointerCapture?.(e.pointerId)
    }
    const move = (e: PointerEvent) => {
      if (!d.active) return
      const dx = e.clientX - d.lastX
      const dy = e.clientY - d.lastY
      d.moved += Math.abs(dx) + Math.abs(dy)
      d.lastX = e.clientX; d.lastY = e.clientY
      d.ry = THREE.MathUtils.clamp(d.ry + dx * 0.004, -0.6, 0.6)
      d.rx = THREE.MathUtils.clamp(d.rx + dy * 0.0025, -0.24, 0.24)
      d.vy = dx * 0.004; d.vx = dy * 0.0025
    }
    const up = () => { d.active = false; el.style.cursor = 'grab' }
    el.style.cursor = 'grab'
    el.style.touchAction = 'pan-y'
    el.addEventListener('pointerdown', down)
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', up)
    return () => {
      el.removeEventListener('pointerdown', down)
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', up)
    }
  }, [gl])

  const fireBurst = (index: number) => {
    burst.current = { index, start: clockT.current }
    onBurst()
  }

  useFrame(({ clock, pointer, camera }) => {
    const t = clock.getElapsedTime()
    clockT.current = t
    elapsed.current = t
    const d = drag.current
    if (!group.current || !tilt.current) return

    // Inertia after release.
    if (!d.active && (Math.abs(d.vy) > 0.0001 || Math.abs(d.vx) > 0.0001)) {
      d.ry = THREE.MathUtils.clamp(d.ry + d.vy, -0.6, 0.6)
      d.rx = THREE.MathUtils.clamp(d.rx + d.vx, -0.24, 0.24)
      d.vy *= 0.94; d.vx *= 0.94
    }
    const idleSway = reducedMotion ? 0 : Math.sin(t * 0.24) * 0.07
    const idleBob = reducedMotion ? 0 : Math.sin(t * 0.45) * 0.06
    group.current.rotation.y += ((-0.14 + d.ry + idleSway) - group.current.rotation.y) * 0.06
    tilt.current.rotation.x += ((-0.1 + d.rx) - tilt.current.rotation.x) * 0.06
    group.current.position.y = idleBob

    // Camera drifts toward the pointer for depth.
    camera.position.x += (pointer.x * 0.55 - camera.position.x) * 0.035
    camera.position.y += (0.15 + pointer.y * 0.3 - camera.position.y) * 0.035
    camera.lookAt(0, 0, 0)

    if (mouseLight.current) {
      mouseLight.current.position.x += (pointer.x * 3 - mouseLight.current.position.x) * 0.08
      mouseLight.current.position.y += (pointer.y * 2 + 1 - mouseLight.current.position.y) * 0.08
    }

    // Traveling pulses along the trend line.
    if (!reducedMotion) {
      pulses.current.forEach((p, k) => {
        if (!p) return
        const u = (t * 0.09 + k * 0.33) % 1
        const pos = trendCurve.getPoint(u)
        p.position.copy(pos)
        const s = 1 + Math.sin(t * 6 + k * 2.1) * 0.18
        p.scale.setScalar(s)
      })
    }

    // Candle breathing + hover pop + burst shockwave.
    const dummy = new THREE.Object3D()
    const b = burst.current
    const age = b ? t - b.start : 99
    const hov = hoveredIdx.current
    const animate = !reducedMotion
    for (let i = 0; i < candles.length; i++) {
      const c = candles[i]
      const breathe = animate ? Math.sin(t * 1.5 + i * 0.45) * 0.05 * (0.5 + c.vol) : 0
      let wave = 0
      if (b && age < 1.4) {
        const dist = Math.abs(i - b.index)
        wave = Math.exp(-dist * 0.32) * Math.exp(-age * 2.4) * Math.sin(age * 11 - dist * 0.7)
      }
      const isHov = hov === i
      dummy.position.set(c.x, c.mid + breathe * 0.4 + wave * 0.3, isHov ? 0.12 : 0)
      dummy.scale.set(isHov ? 1.7 : 1, c.body * (1 + breathe + wave * 0.55) * (isHov ? 1.12 : 1), isHov ? 1.7 : 1)
      dummy.rotation.set(0, 0, 0)
      dummy.updateMatrix()
      bodies.current.setMatrixAt(i, dummy.matrix)
      dummy.position.set(c.x, (c.high + c.low) / 2 + breathe * 0.3 + wave * 0.3, isHov ? 0.1 : -0.012)
      dummy.scale.set(isHov ? 0.5 : 0.24, (c.high - c.low) * (1 + breathe * 0.6 + wave * 0.4), 1)
      dummy.updateMatrix()
      wicks.current.setMatrixAt(i, dummy.matrix)
    }
    bodies.current.instanceMatrix.needsUpdate = true
    wicks.current.instanceMatrix.needsUpdate = true

    // Hover glow disc follows the hovered candle.
    if (hoverGlow.current) {
      if (hov != null) {
        const c = candles[hov]
        hoverGlow.current.visible = true
        hoverGlow.current.position.x += (c.x - hoverGlow.current.position.x) * 0.25
        hoverGlow.current.position.y = c.low - 0.32
        const m = hoverGlow.current.material as THREE.MeshBasicMaterial
        m.opacity += (0.34 - m.opacity) * 0.2
      } else {
        const m = hoverGlow.current.material as THREE.MeshBasicMaterial
        m.opacity *= 0.9
        if (m.opacity < 0.02) hoverGlow.current.visible = false
      }
    }

    // Burst rings expand + fade.
    ;[ringA, ringB].forEach((r, k) => {
      if (!r.current || !b) return
      const local = age - k * 0.14
      if (local < 0 || local > 1.1) { r.current.visible = false; return }
      const c = candles[b.index]
      r.current.visible = true
      r.current.position.set(c.x, c.mid, 0.15)
      const s = 0.25 + (local / 1.1) * (1.6 + k * 0.7)
      r.current.scale.setScalar(s)
      const m = r.current.material as THREE.MeshBasicMaterial
      m.opacity = Math.max(0, 0.65 * (1 - local / 1.1))
    })

    // Diamond markers bob.
    if (animate && diamonds.current) {
      const dm = new THREE.Object3D()
      wideIdx.forEach((ci, k) => {
        const c = candles[ci]
        dm.position.set(c.x, c.high + 0.32 + Math.sin(t * 2 + k * 1.3) * 0.05, 0.02)
        dm.rotation.set(0, t * 0.8 + k, 0)
        dm.scale.setScalar(hov === ci ? 1.7 : 1)
        dm.updateMatrix()
        diamonds.current.setMatrixAt(k, dm.matrix)
      })
      diamonds.current.instanceMatrix.needsUpdate = true
    }

    // Dust drift.
    if (dust.current && animate) {
      dust.current.rotation.y = Math.sin(t * 0.05) * 0.08
      dust.current.position.y = Math.sin(t * 0.3) * 0.08
    }
  })

  return (
    <group ref={group} rotation={[0, -0.14, -0.015]}>
      <group ref={tilt} rotation={[-0.1, 0, 0]}>
        {/* backdrop grid */}
        <lineSegments>
          <bufferGeometry><bufferAttribute attach="attributes-position" args={[new Float32Array(grid.flatMap(p => [p.x, p.y, p.z])), 3]} /></bufferGeometry>
          <lineBasicMaterial color={TEAL_LT} transparent opacity={0.13} />
        </lineSegments>

        {/* range-regime ribbon */}
        <mesh geometry={ribbon}>
          <meshBasicMaterial color={TEAL_LT} transparent opacity={0.09} side={THREE.DoubleSide} depthWrite={false} />
        </mesh>

        {/* wicks + bodies (hoverable) */}
        <instancedMesh
          ref={wicks}
          args={[undefined, undefined, candles.length]}
          onPointerMove={e => { e.stopPropagation(); onHover({ index: e.instanceId!, datum: candles[e.instanceId!] }) }}
          onPointerOut={() => onHover(null)}
          onClick={e => { e.stopPropagation(); if (drag.current.moved < 8 && e.instanceId != null) fireBurst(e.instanceId) }}
        >
          <boxGeometry args={[0.02, 1, 0.02]} /><meshBasicMaterial toneMapped={false} />
        </instancedMesh>
        <instancedMesh
          ref={bodies}
          args={[undefined, undefined, candles.length]}
          onPointerMove={e => { e.stopPropagation(); onHover({ index: e.instanceId!, datum: candles[e.instanceId!] }) }}
          onPointerOut={() => onHover(null)}
          onClick={e => { e.stopPropagation(); if (drag.current.moved < 8 && e.instanceId != null) fireBurst(e.instanceId) }}
        >
          <boxGeometry args={[0.078, 1, 0.075]} /><meshStandardMaterial roughness={0.32} metalness={0.25} emissive={TEAL} emissiveIntensity={0.3} />
        </instancedMesh>

        {/* wide-regime diamond markers */}
        <instancedMesh ref={diamonds} args={[undefined, undefined, wideIdx.length]}>
          <octahedronGeometry args={[0.045]} /><meshBasicMaterial color={EMBER_LT} transparent opacity={0.9} toneMapped={false} />
        </instancedMesh>

        {/* trend tube + traveling pulses */}
        <mesh geometry={trendTube}>
          <meshBasicMaterial color={EMBER} transparent opacity={0.8} />
        </mesh>
        {[0, 1, 2].map(k => (
          <mesh key={k} ref={el => { pulses.current[k] = el }}>
            <sphereGeometry args={[0.05 - k * 0.008, 18, 18]} />
            <meshBasicMaterial color={k === 0 ? '#FFF3E4' : EMBER_LT} toneMapped={false} />
          </mesh>
        ))}

        {/* hover glow + burst rings */}
        <mesh ref={hoverGlow} position={[0, -1.4, -0.1]} rotation={[-Math.PI / 2, 0, 0]} visible={false}>
          <circleGeometry args={[0.3, 40]} />
          <meshBasicMaterial color={EMBER} transparent opacity={0} side={THREE.DoubleSide} depthWrite={false} />
        </mesh>
        {[ringA, ringB].map((r, k) => (
          <mesh key={k} ref={r} visible={false}>
            <ringGeometry args={[0.42, 0.47, 48]} />
            <meshBasicMaterial color={k === 0 ? EMBER_LT : TEAL_LT} transparent opacity={0} side={THREE.DoubleSide} depthWrite={false} />
          </mesh>
        ))}

        {/* dust + floor */}
        <points ref={dust}>
          <bufferGeometry><bufferAttribute attach="attributes-position" args={[dustPos, 3]} /></bufferGeometry>
          <pointsMaterial color={TEAL_LT} size={0.022} transparent opacity={0.55} sizeAttenuation depthWrite={false} />
        </points>
        <mesh position={[0, -1.62, 0]} rotation={[-Math.PI / 2, 0, 0]}>
          <planeGeometry args={[7.2, 3.2]} /><meshBasicMaterial color={TEAL_LT} transparent opacity={0.05} side={THREE.DoubleSide} depthWrite={false} />
        </mesh>

        <pointLight ref={mouseLight} position={[0, 1.5, 3]} intensity={6} distance={9} color="#FFF3E4" />
      </group>
    </group>
  )
}

function PixelRatio() {
  const { gl } = useThree()
  useEffect(() => { gl.setPixelRatio(Math.min(window.devicePixelRatio, 1.7)) }, [gl])
  return null
}

function fmt(n: number) {
  return (100 + n * 22).toFixed(1)
}

export function HeroCanvas({ reducedMotion }: { reducedMotion: boolean }) {
  const [hovered, setHovered] = useState<HoverInfo>(null)
  const [burstKey, setBurstKey] = useState(0)
  return <div className="hp-hero3d">
    <Canvas
      camera={{ position: [0, 0.15, 6.9], fov: 42 }}
      gl={{ antialias: true, alpha: true }}
      style={{ background: 'transparent' }}
      dpr={typeof window !== 'undefined' ? Math.min(window.devicePixelRatio, 1.7) : 1}
    >
      <PixelRatio />
      <ambientLight intensity={0.75} />
      <directionalLight position={[3, 4, 5]} intensity={1.15} color="#FFF6E8" />
      <pointLight position={[2.4, 2.6, 3.4]} intensity={14} distance={14} color={EMBER_LT} />
      <pointLight position={[-3.2, -1.6, 2.4]} intensity={10} distance={12} color={TEAL_LT} />
      <Suspense fallback={null}>
        <VolatilityField
          reducedMotion={reducedMotion}
          hovered={hovered}
          onHover={setHovered}
          onBurst={() => setBurstKey(k => k + 1)}
        />
      </Suspense>
    </Canvas>

    {/* Live readout: follows hover state (pointer-events none so orbit drag passes through). */}
    <div className="hp-3d-readout" aria-live="polite">
      {hovered ? <>
        <span className="hp-3d-readout-top">CANDLE {String(hovered.index + 1).padStart(2, '0')}/{FIELD_N} · {hovered.datum.regime} RANGE</span>
        <span className="hp-3d-readout-grid">
          <span>O {fmt(hovered.datum.open)}</span><span>H {fmt(hovered.datum.high)}</span>
          <span>L {fmt(hovered.datum.low)}</span><span>C {fmt(hovered.datum.close)}</span>
        </span>
        <span className="hp-3d-readout-bar"><i style={{ width: `${Math.min(100, hovered.datum.vol * 160)}%` }} /></span>
      </> : <>
        <span className="hp-3d-readout-top">54 BARS · RANGE-REGIME STUDY</span>
        <span className="hp-3d-readout-idle">Hover a candle to inspect its range — diamonds mark wide-range bars.</span>
      </>}
    </div>
    <div key={burstKey} className={`hp-3d-hint${burstKey > 0 ? ' hp-3d-hint--burst' : ''}`}>
      Drag to orbit · Scroll glides past · Click a candle for a burst
    </div>
  </div>
}

// ─── Evidence spire (closing section) ─────────────────────────────────
// Deliberately NOT candles: 130 stone-like blocks — one per walk-forward
// fold — wound into a rising helix around a glowing research core.
// Drag to spin · hover a block to inspect its fold · click for a sweep.
const FOLD_N = 130
const FOLD_TURNS = 4.25
const FOLD_LANES = [
  { sym: 'HDFCBANK', lift: 13.5 },
  { sym: 'ICICIBANK', lift: 10.7 },
  { sym: 'INFY', lift: 14.1 },
  { sym: 'RELIANCE', lift: 14.3 },
  { sym: 'TCS', lift: 14.8 },
]

export type FoldHover = { index: number; lane: string; window: number; lift: number } | null

function foldPos(i: number) {
  const f = i / (FOLD_N - 1)
  const ang = f * FOLD_TURNS * Math.PI * 2
  const r = 0.84 - f * 0.14
  return new THREE.Vector3(Math.cos(ang) * r, -1.3 + f * 2.6, Math.sin(ang) * r)
}

function foldLift(i: number) {
  const base = FOLD_LANES[Math.floor(i / 26)].lift
  return base + (((i * 13) % 7) - 3) * 0.3
}

function EvidenceSpire({ reducedMotion = false, hovered, onHover, onSweep }: {
  reducedMotion?: boolean; hovered: FoldHover; onHover: (h: FoldHover) => void; onSweep: () => void
}) {
  const group = useRef<THREE.Group>(null!)
  const blocks = useRef<THREE.InstancedMesh>(null!)
  const ring1 = useRef<THREE.Mesh>(null!)
  const ring2 = useRef<THREE.Mesh>(null!)
  const satellite = useRef<THREE.Mesh>(null!)
  const capstone = useRef<THREE.Mesh>(null!)
  const sweepRing = useRef<THREE.Mesh>(null!)
  const hoverDot = useRef<THREE.Mesh>(null!)
  const laneMarks = useRef<THREE.InstancedMesh>(null!)
  const { gl } = useThree()

  const drag = useRef({ active: false, moved: 0, lastX: 0, spin: -0.5, vel: 0 })
  const sweep = useRef<{ start: number } | null>(null)
  const clockT = useRef(0)
  const hoveredIdx = useRef<number | null>(null)
  hoveredIdx.current = hovered?.index ?? null

  const positions = useMemo(() => Array.from({ length: FOLD_N }, (_, i) => foldPos(i)), [])
  const laneAngles = useMemo(() => [0, 26, 52, 78, 104].map(i => (i / (FOLD_N - 1)) * FOLD_TURNS * Math.PI * 2), [])

  useEffect(() => {
    if (!blocks.current || !laneMarks.current) return
    const m = new THREE.Object3D()
    const col = new THREE.Color()
    const teal = new THREE.Color(TEAL)
    const light = new THREE.Color(TEAL_LT)
    positions.forEach((p, i) => {
      m.position.copy(p)
      m.scale.setScalar(1)
      m.rotation.set(0, (i / FOLD_N) * Math.PI, 0)
      m.updateMatrix()
      blocks.current.setMatrixAt(i, m.matrix)
      col.copy(teal).lerp(light, (i / (FOLD_N - 1)) * 0.7)
      if (i === FOLD_N - 1) col.set(EMBER_LT)
      blocks.current.setColorAt(i, col)
    })
    laneAngles.forEach((a, k) => {
      m.position.set(Math.cos(a) * 1.04, -1.55, Math.sin(a) * 1.04)
      m.scale.setScalar(1)
      m.rotation.set(0, 0, 0)
      m.updateMatrix()
      laneMarks.current.setMatrixAt(k, m.matrix)
      laneMarks.current.setColorAt(k, new THREE.Color(EMBER_LT))
    })
    blocks.current.instanceMatrix.needsUpdate = true
    laneMarks.current.instanceMatrix.needsUpdate = true
    if (blocks.current.instanceColor) blocks.current.instanceColor.needsUpdate = true
    if (laneMarks.current.instanceColor) laneMarks.current.instanceColor.needsUpdate = true
  }, [positions, laneAngles])

  useEffect(() => {
    const el = gl.domElement
    const d = drag.current
    const down = (e: PointerEvent) => {
      d.active = true; d.moved = 0; d.lastX = e.clientX; d.vel = 0
      el.style.cursor = 'grabbing'
      el.setPointerCapture?.(e.pointerId)
    }
    const move = (e: PointerEvent) => {
      if (!d.active) return
      const dx = e.clientX - d.lastX
      d.moved += Math.abs(dx)
      d.lastX = e.clientX
      d.spin += dx * 0.007
      d.vel = dx * 0.007
    }
    const up = () => { d.active = false; el.style.cursor = 'grab' }
    el.style.cursor = 'grab'
    el.style.touchAction = 'pan-y'
    el.addEventListener('pointerdown', down)
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', up)
    return () => {
      el.removeEventListener('pointerdown', down)
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', up)
    }
  }, [gl])

  const describe = (i: number): FoldHover => ({
    index: i,
    lane: FOLD_LANES[Math.floor(i / 26)].sym,
    window: (i % 26) + 1,
    lift: foldLift(i),
  })

  useFrame(({ clock, pointer, camera }) => {
    const t = clock.getElapsedTime()
    clockT.current = t
    if (!group.current) return
    const d = drag.current
    if (!d.active) {
      d.spin += (reducedMotion ? 0 : 0.0038) + d.vel
      d.vel *= 0.94
    }
    group.current.rotation.y = d.spin
    if (!reducedMotion) group.current.position.y = Math.sin(t * 0.5) * 0.06

    camera.position.x += (pointer.x * 0.4 - camera.position.x) * 0.04
    camera.lookAt(0, 0.1, 0)

    if (ring1.current) ring1.current.rotation.z = t * 0.28
    if (ring2.current) ring2.current.rotation.z = -t * 0.2
    if (satellite.current) {
      const a = t * 0.7
      satellite.current.position.set(Math.cos(a) * 1.16, Math.sin(t * 0.5) * 0.35, Math.sin(a) * 1.16)
    }
    if (capstone.current && !reducedMotion) {
      capstone.current.rotation.y = t * 0.7
      capstone.current.position.y = 1.56 + Math.sin(t * 1.4) * 0.06
    }

    // Block scales: hover pop + traveling sweep wave.
    const s = sweep.current
    const age = s ? t - s.start : 99
    const sweeping = s && age < 1.4
    const sy = -1.5 + (age / 1.3) * 3.4
    const hov = hoveredIdx.current
    const dummy = new THREE.Object3D()
    for (let i = 0; i < positions.length; i++) {
      const p = positions[i]
      let boost = 0
      if (sweeping) {
        const dy = p.y - sy
        boost = Math.exp(-(dy * dy) / 0.07) * Math.max(0, 1 - age / 1.3)
      }
      const isHov = hov === i
      dummy.position.copy(p)
      dummy.scale.setScalar((isHov ? 1.65 : 1) + boost * 1.15)
      dummy.rotation.set(0, (i / FOLD_N) * Math.PI + (reducedMotion ? 0 : t * 0.15), 0)
      dummy.updateMatrix()
      blocks.current.setMatrixAt(i, dummy.matrix)
    }
    blocks.current.instanceMatrix.needsUpdate = true

    if (sweepRing.current) {
      if (sweeping) {
        sweepRing.current.visible = true
        sweepRing.current.position.y = sy
        const m = sweepRing.current.material as THREE.MeshBasicMaterial
        m.opacity = 0.55 * Math.max(0, 1 - age / 1.3)
      } else sweepRing.current.visible = false
    }
    if (hoverDot.current) {
      if (hov != null) {
        hoverDot.current.visible = true
        hoverDot.current.position.copy(positions[hov])
        const sc = 1 + Math.sin(t * 6) * 0.15
        hoverDot.current.scale.setScalar(sc)
      } else hoverDot.current.visible = false
    }
  })

  return (
    <group ref={group} rotation={[0, -0.5, 0]}>
      {/* research core */}
      <mesh position={[0, 0, 0]}>
        <cylinderGeometry args={[0.014, 0.014, 3.1, 12]} />
        <meshBasicMaterial color={TEAL_LT} transparent opacity={0.55} toneMapped={false} />
      </mesh>
      <mesh position={[0, 0, 0]}>
        <cylinderGeometry args={[0.06, 0.06, 3.0, 12]} />
        <meshBasicMaterial color={TEAL_LT} transparent opacity={0.1} toneMapped={false} depthWrite={false} />
      </mesh>

      {/* 130 fold blocks */}
      <instancedMesh
        ref={blocks}
        args={[undefined, undefined, FOLD_N]}
        onPointerMove={e => { e.stopPropagation(); if (e.instanceId != null) onHover(describe(e.instanceId)) }}
        onPointerOut={() => onHover(null)}
        onClick={e => {
          e.stopPropagation()
          if (drag.current.moved >= 8) return
          if (e.instanceId != null) onHover(describe(e.instanceId))
          sweep.current = { start: clockT.current }
          onSweep()
        }}
      >
        <boxGeometry args={[0.105, 0.105, 0.105]} />
        <meshStandardMaterial roughness={0.38} metalness={0.3} emissive={TEAL} emissiveIntensity={0.28} />
      </instancedMesh>

      {/* capstone: the locked finding */}
      <mesh ref={capstone} position={[0, 1.56, 0]}>
        <octahedronGeometry args={[0.13]} />
        <meshBasicMaterial color={EMBER_LT} toneMapped={false} />
      </mesh>
      <mesh position={[0, 1.56, 0]}>
        <sphereGeometry args={[0.2, 20, 20]} />
        <meshBasicMaterial color={EMBER} transparent opacity={0.16} depthWrite={false} />
      </mesh>

      {/* gyroscope rings + satellite */}
      <mesh ref={ring1} rotation={[Math.PI / 2.35, 0, 0]}>
        <torusGeometry args={[1.16, 0.008, 8, 90]} />
        <meshBasicMaterial color={TEAL_LT} transparent opacity={0.4} toneMapped={false} />
      </mesh>
      <mesh ref={ring2} rotation={[Math.PI / 1.8, 0.4, 0]}>
        <torusGeometry args={[1.32, 0.006, 8, 90]} />
        <meshBasicMaterial color={EMBER_LT} transparent opacity={0.3} toneMapped={false} />
      </mesh>
      <mesh ref={satellite}>
        <sphereGeometry args={[0.035, 14, 14]} />
        <meshBasicMaterial color="#FFF3E4" toneMapped={false} />
      </mesh>

      {/* sweep wave ring + hover marker */}
      <mesh ref={sweepRing} rotation={[Math.PI / 2, 0, 0]} visible={false}>
        <torusGeometry args={[0.95, 0.02, 8, 64]} />
        <meshBasicMaterial color={EMBER_LT} transparent opacity={0} toneMapped={false} depthWrite={false} />
      </mesh>
      <mesh ref={hoverDot} visible={false}>
        <sphereGeometry args={[0.11, 16, 16]} />
        <meshBasicMaterial color={EMBER_LT} transparent opacity={0.35} depthWrite={false} />
      </mesh>

      {/* base: 5 ember ticks = 5 assets */}
      <instancedMesh ref={laneMarks} args={[undefined, undefined, 5]}>
        <octahedronGeometry args={[0.045]} />
        <meshBasicMaterial color={EMBER_LT} toneMapped={false} />
      </instancedMesh>
      <mesh position={[0, -1.58, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <circleGeometry args={[1.06, 48]} />
        <meshBasicMaterial color={TEAL} transparent opacity={0.08} side={THREE.DoubleSide} />
      </mesh>
      <mesh position={[0, -1.58, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[1.04, 1.06, 64]} />
        <meshBasicMaterial color={TEAL_LT} transparent opacity={0.3} side={THREE.DoubleSide} />
      </mesh>
      <mesh position={[0, -1.58, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[1.3, 1.31, 72]} />
        <meshBasicMaterial color={TEAL_LT} transparent opacity={0.14} side={THREE.DoubleSide} />
      </mesh>
    </group>
  )
}

export function ClosingCanvas({ reducedMotion }: { reducedMotion: boolean }) {
  const [hovered, setHovered] = useState<FoldHover>(null)
  const [sweepKey, setSweepKey] = useState(0)
  return <div className="hp-closing3d">
    <Canvas
      camera={{ position: [0, 0.2, 5.8], fov: 40 }}
      gl={{ antialias: true, alpha: true }}
      style={{ background: 'transparent' }}
      dpr={typeof window !== 'undefined' ? Math.min(window.devicePixelRatio, 1.7) : 1}
    >
      <PixelRatio />
      <ambientLight intensity={0.9} />
      <directionalLight position={[3, 4, 5]} intensity={1.1} color="#FFF6E8" />
      <pointLight position={[-3, -1, 2]} intensity={1.2} color="#78A99F" />
      <pointLight position={[0, 1.9, 1.2]} intensity={4} distance={6} color="#F05A40" />
      <Suspense fallback={null}>
        <EvidenceSpire
          reducedMotion={reducedMotion}
          hovered={hovered}
          onHover={setHovered}
          onSweep={() => setSweepKey(k => k + 1)}
        />
      </Suspense>
    </Canvas>
    <div className="hp-close-readout" aria-live="polite">
      {hovered
        ? <>FOLD {String(hovered.index + 1).padStart(3, '0')}/{FOLD_N} · {hovered.lane} · W{hovered.window} · +{hovered.lift.toFixed(1)}pp</>
        : <>130/130 folds beat baseline — hover a block</>}
    </div>
    <div key={sweepKey} className={`hp-close-hint${sweepKey > 0 ? ' hp-close-hint--sweep' : ''}`}>
      Drag to spin · Click for a validation sweep
    </div>
  </div>
}
