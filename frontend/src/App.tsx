import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { ProtectedRoute } from './components/ProtectedRoute'
import { Login } from './pages/Login'
import { Pulsecast } from './pages/Pulsecast'
import { paths } from './routes/paths'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path={paths.login} element={<Login />} />
        <Route path="/" element={<Navigate to={paths.dashboard} replace />} />
        <Route
          path="/:screen"
          element={
            <ProtectedRoute>
              <Pulsecast />
            </ProtectedRoute>
          }
        />
        <Route path="*" element={<Navigate to={paths.dashboard} replace />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
