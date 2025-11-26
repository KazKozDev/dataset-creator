import axios from 'axios';

// Configure base API
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Domain endpoints
export const getDomains = async () => {
  const response = await api.get('/domains');
  return response.data;
};

export const getDomainDetails = async (domainKey) => {
  const response = await api.get(`/domains/${domainKey}`);
  return response.data;
};

export const getSubdomains = async (domainKey) => {
  const response = await api.get(`/domains/${domainKey}/subdomains`);
  return response.data;
};

export const getSubdomainDetails = async (domainKey, subdomainKey) => {
  const response = await api.get(`/domains/${domainKey}/subdomains/${subdomainKey}`);
  return response.data;
};

export const getCommonParams = async () => {
  const response = await api.get('/common-params');
  return response.data;
};

export const getModels = async (provider) => {
  const response = await api.get('/providers/models', {
    params: { provider },
  });
  return response.data;
};

export const setProviderConfig = async (config) => {
  const response = await api.post('/providers/config', config);
  return response.data;
};

// Generator endpoints
export const startGenerator = async (params) => {
  const response = await api.post('/generator/start', params);
  return response.data;
};

export const getGeneratorStatus = async (jobId) => {
  const response = await api.get(`/generator/status/${jobId}`);
  return response.data;
};

export const cancelGenerator = async (jobId) => {
  const response = await api.post(`/generator/cancel/${jobId}`);
  return response.data;
};

// Quality control endpoints
export const startQualityControl = async (params) => {
  const response = await api.post('/quality/start', params);
  return response.data;
};

export const getQualityStatus = async (jobId) => {
  const response = await api.get(`/quality/status/${jobId}`);
  return response.data;
};

export const cancelQuality = async (jobId) => {
  const response = await api.post(`/quality/cancel/${jobId}`);
  return response.data;
};

// Dataset endpoints
export const getDatasets = async (params) => {
  const response = await api.get('/datasets', { params });
  return response.data;
};

export const getDatasetDetails = async (datasetId) => {
  console.log(`Fetching dataset with ID: ${datasetId}`);
  try {
    const response = await api.get(`/datasets/${datasetId}`);
    console.log('Dataset details response:', response.data);
    return response.data;
  } catch (error) {
    console.error('Error fetching dataset details:', error);
    throw error;
  }
};

export const getDatasetExamples = async (datasetId, params) => {
  console.log(`Fetching examples for dataset ID: ${datasetId}, params:`, params);
  try {
    const response = await api.get(`/datasets/${datasetId}/examples`, { params });
    console.log('Dataset examples response:', response.data);
    return response.data;
  } catch (error) {
    console.error('Error fetching dataset examples:', error);
    throw error;
  }
};

export const getDatasetStats = async (datasetId) => {
  const response = await api.get(`/datasets/${datasetId}/stats`);
  return response.data;
};

export const downloadDataset = async (datasetId) => {
  const response = await api.get(`/datasets/${datasetId}/download`, {
    responseType: 'blob',
  });
  return response.data;
};

export const exportDatasetToCsv = async (datasetId) => {
  const response = await api.post(`/datasets/${datasetId}/export-csv`);
  return response.data;
};

export const deleteDataset = async (datasetId) => {
  const response = await api.delete(`/datasets/${datasetId}`);
  return response.data;
};

export const uploadDataset = async (file, params) => {
  const formData = new FormData();
  formData.append('file', file);

  // Add additional params to form data
  Object.entries(params).forEach(([key, value]) => {
    if (value) formData.append(key, value);
  });

  const response = await api.post('/datasets/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
};

export const mergeDatasets = async (datasetIds, name, description) => {
  const response = await api.post('/datasets/merge', {
    dataset_ids: datasetIds,
    name,
    description,
  });
  return response.data;
};

export const sampleDataset = async (datasetId, count, name) => {
  const response = await api.post('/datasets/sample', {
    dataset_id: datasetId,
    count,
    name,
  });
  return response.data;
};

export const convertDatasetFormat = async (datasetId, toFormat, name) => {
  const response = await api.post('/datasets/convert', {
    dataset_id: datasetId,
    to_format: toFormat,
    name,
  });
  return response.data;
};

// Utility endpoints
export const scanForDatasets = async () => {
  const response = await api.get('/utils/scan-datasets');
  return response.data;
};

// Task endpoints
export const getAllTasks = async () => {
  const response = await api.get('/tasks');
  return response.data;
};

export const getTaskDetails = async (taskId) => {
  const response = await api.get(`/tasks/${taskId}`);
  return response.data;
};

export const cancelTask = async (taskId) => {
  const response = await api.post(`/tasks/${taskId}/cancel`);
  return response.data;
};

export const deleteTask = async (taskId) => {
  const response = await api.delete(`/tasks/${taskId}`);
  return response.data;
};

// System settings endpoints
export const getSystemSettings = async () => {
  const response = await api.get('/settings');
  return response.data;
};

export const updateSystemSettings = async (settings) => {
  const response = await api.post('/settings', settings);
  return response.data;
};

export const getSystemStatus = async () => {
  const response = await api.get('/settings/status');
  return response.data;
};

export const clearCache = async () => {
  const response = await api.post('/utils/clear-cache');
  return response.data;
};

export const updateExample = async (datasetId, exampleId, content) => {
  console.log('Updating example:', { datasetId, exampleId, content });
  try {
    const response = await fetch(`${API_BASE_URL}/api/datasets/${datasetId}/examples/${exampleId}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ content }),
    });

    console.log('Update response status:', response.status);
    const data = await response.json();
    console.log('Update response data:', data);

    if (!response.ok) {
      throw new Error(data.detail || 'Failed to update example');
    }

    return data;
  } catch (error) {
    console.error('Error updating example:', error);
    throw error;
  }
};

// Prompts endpoints
export const getPrompts = async () => {
  const response = await api.get('/prompts');
  return response.data;
};

export const getPrompt = async (promptId) => {
  const response = await api.get(`/prompts/${promptId}`);
  return response.data;
};

export const createPrompt = async (promptData) => {
  const response = await api.post('/prompts', promptData);
  return response.data;
};

export const updatePrompt = async (promptId, promptData) => {
  const response = await api.put(`/prompts/${promptId}`, promptData);
  return response.data;
};

export const deletePrompt = async (promptId) => {
  const response = await api.delete(`/prompts/${promptId}`);
  return response.data;
};

// Reviews endpoints
export const getReviews = async (datasetId) => {
  const response = await api.get(`/datasets/${datasetId}/reviews`);
  return response.data;
};

export const createReview = async (reviewData) => {
  const response = await api.post('/reviews', reviewData);
  return response.data;
};

// Comments endpoints
export const getComments = async (datasetId) => {
  const response = await api.get(`/datasets/${datasetId}/comments`);
  return response.data;
};

export const createComment = async (commentData) => {
  const response = await api.post('/comments', commentData);
  return response.data;
};

// Versions endpoints
export const getVersions = async (datasetId) => {
  const response = await api.get(`/datasets/${datasetId}/versions`);
  return response.data;
};

export const createVersion = async (versionData) => {
  const response = await api.post('/versions', versionData);
  return response.data;
};

// Search examples
export const searchExamples = async (datasetId, query, params = {}) => {
  const response = await api.get(`/datasets/${datasetId}/examples/search`, {
    params: { q: query, ...params }
  });
  return response.data;
};

// Template API
export const getTemplates = async (domain) => {
  const params = domain ? { domain } : {};
  const response = await api.get('/templates', { params });
  return response.data;
};

export const createTemplate = async (templateData) => {
  const response = await api.post('/templates', templateData);
  return response.data;
};

export const updateTemplate = async (id, templateData) => {
  const response = await api.put(`/templates/${id}`, templateData);
  return response.data;
};

export const deleteTemplate = async (id) => {
  const response = await api.delete(`/templates/${id}`);
  return response.data;
};

export const getTemplateVersions = async (id) => {
  const response = await api.get(`/templates/${id}/versions`);
  return response.data;
};

export const restoreTemplateVersion = async (templateId, versionId) => {
  const response = await api.post(`/templates/${templateId}/restore/${versionId}`);
  return response.data;
};

// Analytics endpoints
export const getAnalyticsSummary = async (days = 30) => {
  const response = await api.get('/analytics/summary', {
    params: { days }
  });
  return response.data;
};

// Model management endpoints
export const getProviders = async () => {
  const response = await api.get('/models/providers');
  return response.data;
};

export const getProviderModels = async (provider) => {
  const response = await api.get(`/models/${provider}`);
  return response.data;
};

export const getModelInfo = async (provider, modelId) => {
  const response = await api.get(`/models/${provider}/${modelId}/info`);
  return response.data;
};

export const estimateCost = async (provider, model, examplesCount, avgTokens = 1000) => {
  const response = await api.get('/analytics/cost-estimate', {
    params: {
      provider,
      model,
      examples_count: examplesCount,
      avg_tokens_per_example: avgTokens
    }
  });
  return response.data;
};

export default api;