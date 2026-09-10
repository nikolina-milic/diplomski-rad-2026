import { useEffect, useRef, type ReactNode } from 'react'

/**
 * Modal za detalje odluke.
 *
 * Analitičar ga otvara i zatvara desetine puta u smjeni, pa mora raditi i bez
 * miša: Esc zatvara, fokus ulazi u dijalog i vraća se na red sa kojeg je
 * otvoren, a Tab ostaje zarobljen unutra dok je otvoren.
 */
export default function Modal({
  open, onClose, title, children,
}: {
  open: boolean
  onClose: () => void
  title: string
  children: ReactNode
}) {
  const dialogRef = useRef<HTMLDivElement>(null)
  const openerRef = useRef<HTMLElement | null>(null)

  useEffect(() => {
    if (!open) return
    openerRef.current = document.activeElement as HTMLElement | null
    dialogRef.current?.focus()

    // Pozadina ne smije da se skroluje dok je dijalog otvoren.
    const prevOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'

    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { onClose(); return }
      if (e.key !== 'Tab') return
      const focusable = dialogRef.current?.querySelectorAll<HTMLElement>(
        'a[href], button:not(:disabled), input:not(:disabled), select, textarea, [tabindex]:not([tabindex="-1"])',
      )
      if (!focusable || focusable.length === 0) return
      const first = focusable[0]
      const last = focusable[focusable.length - 1]
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault(); last.focus()
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault(); first.focus()
      }
    }
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('keydown', onKey)
      document.body.style.overflow = prevOverflow
      openerRef.current?.focus()
    }
  }, [open, onClose])

  if (!open) return null

  return (
    <div className="modal-backdrop" onMouseDown={(e) => {
      // zatvara samo klik na pozadinu, ne i povlačenje selekcije iz dijaloga
      if (e.target === e.currentTarget) onClose()
    }}>
      <div className="modal" role="dialog" aria-modal="true" aria-label={title}
        ref={dialogRef} tabIndex={-1}>
        <div className="modal-head">
          <div className="eyebrow">{title}</div>
          <button className="modal-x" onClick={onClose} aria-label="Zatvori">×</button>
        </div>
        <div className="modal-body">{children}</div>
      </div>
    </div>
  )
}
