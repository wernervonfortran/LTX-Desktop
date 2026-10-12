import { useEffect, useState } from 'react'
import { useAppSettings } from '../contexts/AppSettingsContext'
import { ApiClient } from '../lib/api-client'

// OpenRouter fork: hosted image models for the GenSpace image picker.
export interface ImageModelSpecItem {
  model: string // "or:<model-id>"
  or_model_id: string
  display_name: string
  price_per_image?: number | null
  supported_aspect_ratios?: string[]
  supports_resolution?: boolean
  supports_edit?: boolean
}

interface ImageGenerationModelSpecsState {
  modelSpecs: ImageModelSpecItem[]
  isLoading: boolean
  errorMessage: string | null
}

export function useImageGenerationModelSpecs(): ImageGenerationModelSpecsState {
  const { settings, modelsVersion } = useAppSettings()
  const hasOpenRouterApiKey = settings.hasOpenRouterApiKey
  const [state, setState] = useState<ImageGenerationModelSpecsState>({
    modelSpecs: [],
    isLoading: false,
    errorMessage: null,
  })

  useEffect(() => {
    if (!hasOpenRouterApiKey) {
      setState({ modelSpecs: [], isLoading: false, errorMessage: null })
      return
    }
    const abortController = new AbortController()
    let isActive = true
    setState((prev) => ({ ...prev, isLoading: true }))

    void (async () => {
      const result = await ApiClient.getGenerateImageModelSpecs(undefined, {
        signal: abortController.signal,
      })
      if (!isActive) return
      if (result.ok) {
        setState({
          modelSpecs: result.data.openrouter_models ?? [],
          isLoading: false,
          errorMessage: null,
        })
        return
      }
      setState({ modelSpecs: [], isLoading: false, errorMessage: result.error.message })
    })()

    return () => {
      abortController.abort()
    }
  }, [hasOpenRouterApiKey, modelsVersion])

  return state
}
