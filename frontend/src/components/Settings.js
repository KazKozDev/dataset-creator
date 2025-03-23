import React, { useState } from 'react';
import {
  Box,
  Button,
  Card,
  CardBody,
  CardHeader,
  FormControl,
  FormLabel,
  Heading,
  Input,
  Select,
  Text,
  useToast,
  VStack,
  Divider,
  SimpleGrid,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  HStack,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  NumberIncrementStepper,
  NumberDecrementStepper,
  Switch,
  useColorModeValue
} from '@chakra-ui/react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { FiRefreshCw, FiTrash2 } from 'react-icons/fi';
import { 
  getProviders, 
  getModels, 
  getSystemSettings, 
  getSystemStatus, 
  setProviderConfig, 
  updateSystemSettings, 
  clearCache 
} from '../services/api';

const Settings = () => {
  const [modelSettings, setModelSettings] = useState({
    provider: 'ollama',
    model: 'gemma:7b',
    apiKey: '',
    baseUrl: 'http://host.docker.internal:11434'
  });
  
  const [systemSettings, setSystemSettings] = useState({
    dataDirectory: '/app/data',
    maxConcurrentJobs: 2,
    enableCaching: true,
    cacheTTL: 24,
    logLevel: 'info'
  });

  const toast = useToast();
  const queryClient = useQueryClient();
  const cardBg = useColorModeValue('white', 'gray.700');
  const borderColor = useColorModeValue('gray.200', 'gray.600');

  // Fetch available providers
  const { data: providersData } = useQuery({
    queryKey: ['providers'],
    queryFn: getProviders
  });

  // Fetch available models for selected provider
  const { data: modelsData } = useQuery({
    queryKey: ['models', modelSettings.provider],
    queryFn: () => getModels(modelSettings.provider),
    enabled: !!modelSettings.provider
  });

  // Fetch current system settings
  useQuery({
    queryKey: ['systemSettings'],
    queryFn: getSystemSettings,
    onSuccess: (data) => {
      if (data.modelSettings) {
        setModelSettings({
          provider: data.modelSettings.provider || 'ollama',
          model: data.modelSettings.model || 'gemma:7b',
          apiKey: data.modelSettings.apiKey || '',
          baseUrl: data.modelSettings.baseUrl || 'http://host.docker.internal:11434'
        });
      }
      
      if (data.systemSettings) {
        setSystemSettings({
          dataDirectory: data.systemSettings.dataDirectory || '/app/data',
          maxConcurrentJobs: data.systemSettings.maxConcurrentJobs || 2,
          enableCaching: data.systemSettings.enableCaching !== false,
          cacheTTL: data.systemSettings.cacheTTL || 24,
          logLevel: data.systemSettings.logLevel || 'info'
        });
      }
    }
  });

  // Fetch system status
  const { data: systemStatus, refetch: refetchStatus } = useQuery({
    queryKey: ['systemStatus'],
    queryFn: getSystemStatus,
    refetchInterval: 30000 // Refresh every 30 seconds
  });

  // Update provider settings mutation
  const updateProviderMutation = useMutation({
    mutationFn: (settings) => setProviderConfig(settings),
    onSuccess: () => {
      toast({
        title: 'Model settings updated',
        status: 'success',
        duration: 3000,
        isClosable: true
      });
      queryClient.invalidateQueries(['systemSettings']);
      queryClient.invalidateQueries(['models', modelSettings.provider]);
    },
    onError: (error) => {
      toast({
        title: 'Error updating model settings',
        description: error.response?.data?.detail || error.message,
        status: 'error',
        duration: 3000,
        isClosable: true
      });
    }
  });

  // Update system settings mutation
  const updateSystemSettingsMutation = useMutation({
    mutationFn: (settings) => updateSystemSettings(settings),
    onSuccess: () => {
      toast({
        title: 'System settings updated',
        status: 'success',
        duration: 3000,
        isClosable: true
      });
      queryClient.invalidateQueries(['systemSettings']);
      refetchStatus();
    },
    onError: (error) => {
      toast({
        title: 'Error updating system settings',
        description: error.response?.data?.detail || error.message,
        status: 'error',
        duration: 3000,
        isClosable: true
      });
    }
  });

  // Clear cache mutation
  const clearCacheMutation = useMutation({
    mutationFn: clearCache,
    onSuccess: () => {
      toast({
        title: 'Cache cleared successfully',
        status: 'success',
        duration: 3000,
        isClosable: true
      });
      refetchStatus();
    },
    onError: (error) => {
      toast({
        title: 'Error clearing cache',
        description: error.response?.data?.detail || error.message,
        status: 'error',
        duration: 3000,
        isClosable: true
      });
    }
  });

  // Handle model settings change
  const handleModelSettingsChange = (field, value) => {
    setModelSettings({
      ...modelSettings,
      [field]: value
    });
  };

  // Handle system settings change
  const handleSystemSettingsChange = (field, value) => {
    setSystemSettings({
      ...systemSettings,
      [field]: value
    });
  };

  // Save model settings
  const saveModelSettings = () => {
    updateProviderMutation.mutate(modelSettings);
  };

  // Save system settings
  const saveSystemSettings = () => {
    updateSystemSettingsMutation.mutate(systemSettings);
  };

  // Format bytes to human readable format
  const formatBytes = (bytes, decimals = 2) => {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
  };

  // Get available providers
  const providers = providersData?.providers || ['ollama', 'openai', 'anthropic'];

  // Get available models
  const models = modelsData?.models || [];

  return (
    <Box>
      <Heading size="lg" mb={6}>Settings</Heading>

      {/* System Status */}
      <Card bg={cardBg} borderWidth="1px" borderColor={borderColor} mb={6} boxShadow="sm">
        <CardHeader>
          <Heading size="md">System Status</Heading>
        </CardHeader>
        <Divider />
        <CardBody>
          <SimpleGrid columns={{ base: 1, md: 3 }} spacing={6}>
            <Stat>
              <StatLabel>Disk Usage</StatLabel>
              <StatNumber>
                {systemStatus?.diskUsage ? 
                  formatBytes(systemStatus.diskUsage.used) : 'Loading...'}
              </StatNumber>
              <StatHelpText>
                {systemStatus?.diskUsage ? 
                  `${Math.round(systemStatus.diskUsage.percentUsed)}% of ${formatBytes(systemStatus.diskUsage.total)}` : ''}
              </StatHelpText>
            </Stat>
            
            <Stat>
              <StatLabel>Memory Usage</StatLabel>
              <StatNumber>
                {systemStatus?.memoryUsage ? 
                  formatBytes(systemStatus.memoryUsage.used) : 'Loading...'}
              </StatNumber>
              <StatHelpText>
                {systemStatus?.memoryUsage ? 
                  `${Math.round(systemStatus.memoryUsage.percentUsed)}% of ${formatBytes(systemStatus.memoryUsage.total)}` : ''}
              </StatHelpText>
            </Stat>
            
            <Stat>
              <StatLabel>Cache Size</StatLabel>
              <StatNumber>
                {systemStatus?.cacheSize ? 
                  formatBytes(systemStatus.cacheSize) : 'Loading...'}
              </StatNumber>
              <HStack spacing={2} mt={2}>
                <Button 
                  size="xs" 
                  leftIcon={<FiRefreshCw />} 
                  onClick={refetchStatus}
                >
                  Refresh
                </Button>
                <Button 
                  size="xs" 
                  colorScheme="red" 
                  leftIcon={<FiTrash2 />} 
                  onClick={() => clearCacheMutation.mutate()}
                  isLoading={clearCacheMutation.isPending}
                >
                  Clear Cache
                </Button>
              </HStack>
            </Stat>
          </SimpleGrid>
        </CardBody>
      </Card>

      <SimpleGrid columns={{ base: 1, md: 2 }} spacing={6}>
        {/* Model Settings */}
        <Card bg={cardBg} borderWidth="1px" borderColor={borderColor} boxShadow="sm">
          <CardHeader>
            <Heading size="md">Model Settings</Heading>
          </CardHeader>
          <Divider />
          <CardBody>
            <VStack spacing={4} align="stretch">
              <FormControl>
                <FormLabel>Provider</FormLabel>
                <Select
                  value={modelSettings.provider}
                  onChange={(e) => handleModelSettingsChange('provider', e.target.value)}
                >
                  {providers.map((provider) => (
                    <option key={provider} value={provider}>
                      {provider.charAt(0).toUpperCase() + provider.slice(1)}
                    </option>
                  ))}
                </Select>
              </FormControl>

              <FormControl>
                <FormLabel>Model</FormLabel>
                <Select
                  value={modelSettings.model}
                  onChange={(e) => handleModelSettingsChange('model', e.target.value)}
                >
                  {modelSettings.provider === 'openai' ? (
                    <>
                      <option value="gpt-4o">GPT-4o</option>
                      <option value="03-mini-high">03-mini-high</option>
                    </>
                  ) : modelSettings.provider === 'anthropic' ? (
                    <>
                      <option value="claude-3-7-sonnet-20250219">Claude 3.7 Sonnet</option>
                    </>
                  ) : models.length > 0 ? (
                    models.map((model) => (
                      <option key={model} value={model}>
                        {model}
                      </option>
                    ))
                  ) : (
                    <option value={modelSettings.model}>{modelSettings.model}</option>
                  )}
                </Select>
              </FormControl>

              {(modelSettings.provider === 'openai' || modelSettings.provider === 'anthropic') && (
                <FormControl>
                  <FormLabel>API Key</FormLabel>
                  <Input
                    type="password"
                    value={modelSettings.apiKey}
                    onChange={(e) => handleModelSettingsChange('apiKey', e.target.value)}
                    placeholder="Enter your API key"
                  />
                </FormControl>
              )}

              <FormControl>
                <FormLabel>Base URL</FormLabel>
                <Input
                  value={modelSettings.baseUrl}
                  onChange={(e) => handleModelSettingsChange('baseUrl', e.target.value)}
                  placeholder="Enter base URL for API"
                />
                <Text fontSize="sm" color="gray.500" mt={1}>
                  For Ollama, use http://host.docker.internal:11434 to connect to local Ollama server
                </Text>
              </FormControl>

              <Button
                colorScheme="blue"
                onClick={saveModelSettings}
                isLoading={updateProviderMutation.isPending}
                mt={2}
                width="200px"
                alignSelf="flex-start"
              >
                Save Model Settings
              </Button>
            </VStack>
          </CardBody>
        </Card>

        {/* System Settings */}
        <Card bg={cardBg} borderWidth="1px" borderColor={borderColor} boxShadow="sm">
          <CardHeader>
            <Heading size="md">System Settings</Heading>
          </CardHeader>
          <Divider />
          <CardBody>
            <VStack spacing={4} align="stretch">
              <FormControl>
                <FormLabel>Data Directory</FormLabel>
                <Input
                  value={systemSettings.dataDirectory}
                  onChange={(e) => handleSystemSettingsChange('dataDirectory', e.target.value)}
                  placeholder="/app/data"
                />
              </FormControl>

              <FormControl>
                <FormLabel>Max Concurrent Jobs</FormLabel>
                <NumberInput
                  value={systemSettings.maxConcurrentJobs}
                  onChange={(value) => handleSystemSettingsChange('maxConcurrentJobs', parseInt(value))}
                  min={1}
                  max={10}
                >
                  <NumberInputField />
                  <NumberInputStepper>
                    <NumberIncrementStepper />
                    <NumberDecrementStepper />
                  </NumberInputStepper>
                </NumberInput>
                <Text fontSize="sm" color="gray.500" mt={1}>
                  Maximum number of jobs that can run simultaneously
                </Text>
              </FormControl>

              <FormControl display="flex" alignItems="center">
                <Switch
                  isChecked={systemSettings.enableCaching}
                  onChange={(e) => handleSystemSettingsChange('enableCaching', e.target.checked)}
                  mr={3}
                />
                <FormLabel mb="0">Enable Caching</FormLabel>
              </FormControl>

              {systemSettings.enableCaching && (
                <FormControl>
                  <FormLabel>Cache TTL (hours)</FormLabel>
                  <NumberInput
                    value={systemSettings.cacheTTL}
                    onChange={(value) => handleSystemSettingsChange('cacheTTL', parseInt(value))}
                    min={1}
                    max={168}
                  >
                    <NumberInputField />
                    <NumberInputStepper>
                      <NumberIncrementStepper />
                      <NumberDecrementStepper />
                    </NumberInputStepper>
                  </NumberInput>
                  <Text fontSize="sm" color="gray.500" mt={1}>
                    Time to live for cached items (in hours)
                  </Text>
                </FormControl>
              )}

              <FormControl>
                <FormLabel>Log Level</FormLabel>
                <Select
                  value={systemSettings.logLevel}
                  onChange={(e) => handleSystemSettingsChange('logLevel', e.target.value)}
                >
                  <option value="debug">Debug</option>
                  <option value="info">Info</option>
                  <option value="warning">Warning</option>
                  <option value="error">Error</option>
                </Select>
              </FormControl>

              <Button
                colorScheme="blue"
                onClick={saveSystemSettings}
                isLoading={updateSystemSettingsMutation.isPending}
                mt={2}
                width="200px"
                alignSelf="flex-start"
              >
                Save System Settings
              </Button>
            </VStack>
          </CardBody>
        </Card>
      </SimpleGrid>
    </Box>
  );
};

export default Settings;
