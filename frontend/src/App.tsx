import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import LiveMonitor from './pages/LiveMonitor'
import Events from './pages/Events'
import ReviewQueue from './pages/ReviewQueue'
import Metrics from './pages/Metrics'
import Config from './pages/Config'
import Registry from './pages/Registry'

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<LiveMonitor />} />
        <Route path="dogadjaji" element={<Events />} />
        <Route path="review" element={<ReviewQueue />} />
        <Route path="metrics" element={<Metrics />} />
        <Route path="config" element={<Config />} />
        <Route path="registry" element={<Registry />} />
      </Route>
    </Routes>
  )
}
