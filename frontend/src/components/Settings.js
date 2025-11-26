import React, { useState, useEffect } from 'react';
import {
  Box,
  Heading,
  VStack,
  HStack,
  Card,
  CardHeader,
  CardBody,
  FormControl,
  FormLabel,
  Input,
  Button,
  Text,
  useToast,
  Divider,
  Badge,
  IconButton,
  InputGroup,
  InputRightElement,
  Select,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  NumberIncrementStepper,
  NumberDecrementStepper,
  Switch,
  SimpleGrid,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  Code,
  useColorModeValue,
} from '@chakra-ui/react';
import { FiEye, FiEyeOff, FiSave, FiPlus, FiTrash2, FiRefreshCw } from 'react-icons/fi';
import { useQuery, useMutation } from '@tanstack/react-query';

const Settings = () => {
  const toast = useToast();
  const cardBg = useColorModeValue('white', 'gray.700');
  const borderColor = useColorModeValue('gray.200', 'gray.600');

  const [apiKeys, setApiKeys] = useState({
    openai: '',
    anthropic: '',
    google: '',
    mistral: '',
  });

  const [showKeys, setShowKeys] = useState({
    openai: false,
    anthropic: false,
    google: false,
    mistral: false,
  });

  const [customModels, setCustomModels] = useState({
    openai: [],
    anthropic: [],
    google: [],
    mistral: [],
  });

  const [newModel, setNewModel] = useState('');
  const [selectedProvider, setSelectedProvider] = useState('openai');

  const [systemSettings, setSystemSettings] = useState({
    dataDirectory: '/app/data',
    maxConcurrentJobs: 3,
    enableCaching: true,
    cacheTTL: 24,
    logLevel: 'info',
  });

  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

  // Fetch system status
  const { data: systemStatus, refetch: refetchStatus } = useQuery({
    queryKey: ['systemStatus'],
    queryFn: async () => {
      const response = await fetch(`${API_BASE_URL}/settings/status`);
      return response.json();
    },
  });

  useEffect(() => {
    // Load from localStorage
    const savedKeys = localStorage.getItem('llm_api_keys');
    const savedModels = localStorage.getItem('custom_models');
    const savedSystem = localStorage.getItem('system_settings');

    if (savedKeys) setApiKeys(JSON.parse(savedKeys));
    if (savedModels) setCustomModels(JSON.parse(savedModels));
    if (savedSystem) setSystemSettings(JSON.parse(savedSystem));
  }, []);

  const handleSaveApiKey = (provider) => {
    localStorage.setItem('llm_api_keys', JSON.stringify(apiKeys));
    
    fetch(`${API_BASE_URL}/settings/api-key`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        provider: provider.toUpperCase(),
        api_key: apiKeys[provider],
      }),
    })
      .then((response) => {
        if (!response.ok) throw new Error('Failed to save API key');
        return response.json();
      })
      .then(() => {
        toast({
          title: 'API Key Saved',
          description: `${provider.toUpperCase()} API key saved`,
          status: 'success',
          duration: 2000,
        });
      })
      .catch((error) => {
        toast({
          title: 'Error',
          description: error.message,
          status: 'error',
          duration: 3000,
        });
      });
  };

  const handleAddModel = () => {
    if (!newModel.trim()) return;

    const updated = {
      ...customModels,
      [selectedProvider]: [...customModels[selectedProvider], newModel.trim()],
    };

    setCustomModels(updated);
    localStorage.setItem('custom_models', JSON.stringify(updated));
    setNewModel('');

    toast({
      title: 'Model Added',
      status: 'success',
      duration: 2000,
    });
  };

  const handleRemoveModel = (provider, modelName) => {
    const updated = {
      ...customModels,
      [provider]: customModels[provider].filter((m) => m !== modelName),
    };

    setCustomModels(updated);
    localStorage.setItem('custom_models', JSON.stringify(updated));
  };

  const saveSystemSettings = () => {
    localStorage.setItem('system_settings', JSON.stringify(systemSettings));
    toast({
      title: 'System Settings Saved',
      status: 'success',
      duration: 2000,
    });
  };

  const clearCacheMutation = useMutation({
    mutationFn: async () => {
      const response = await fetch(`${API_BASE_URL}/settings/clear-cache`, { method: 'POST' });
      return response.json();
    },
    onSuccess: () => {
      toast({
        title: 'Cache Cleared',
        status: 'success',
        duration: 2000,
      });
      refetchStatus();
    },
  });

  const formatBytes = (bytes) => {
    if (!bytes) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${(bytes / Math.pow(k, i)).toFixed(2)} ${sizes[i]}`;
  };

  return (
    <Box>
      <Heading size="md" mb={6}>Settings</Heading>

      {/* System Status */}
      <Card bg={cardBg} borderWidth="1px" borderColor={borderColor} mb={6}>
        <CardHeader>
          <Heading size="md">System Status</Heading>
        </CardHeader>
        <Divider />
        <CardBody>
          <SimpleGrid columns={{ base: 1, md: 3 }} spacing={6}>
            <Stat>
              <StatLabel>Active Jobs</StatLabel>
              <StatNumber>{systemStatus?.activeJobs || 0}</StatNumber>
              <StatHelpText>Currently running</StatHelpText>
            </Stat>
            <Stat>
              <StatLabel>Total Datasets</StatLabel>
              <StatNumber>{systemStatus?.totalDatasets || 0}</StatNumber>
              <StatHelpText>Created</StatHelpText>
            </Stat>
            <Stat>
              <StatLabel>Cache Size</StatLabel>
              <StatNumber>{formatBytes(systemStatus?.cacheSize)}</StatNumber>
              <HStack spacing={2} mt={2}>
                <Button size="xs" leftIcon={<FiRefreshCw />} onClick={refetchStatus}>
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
        {/* API Keys & Models */}
        <Card bg={cardBg} borderWidth="1px" borderColor={borderColor}>
          <CardHeader>
            <Heading size="md">API Keys & Models</Heading>
          </CardHeader>
          <Divider />
          <CardBody>
            <VStack spacing={4} align="stretch">
              {/* API Keys */}
              {['openai', 'anthropic', 'google', 'mistral'].map((provider) => (
                <FormControl key={provider}>
                  <FormLabel>
                    {provider.charAt(0).toUpperCase() + provider.slice(1)} API Key
                    {apiKeys[provider] && <Badge ml={2} colorScheme="green">Set</Badge>}
                  </FormLabel>
                  <HStack>
                    <InputGroup size="sm">
                      <Input
                        type={showKeys[provider] ? 'text' : 'password'}
                        value={apiKeys[provider]}
                        onChange={(e) => setApiKeys({ ...apiKeys, [provider]: e.target.value })}
                        placeholder={`Enter ${provider} API key`}
                      />
                      <InputRightElement>
                        <IconButton
                          size="xs"
                          variant="ghost"
                          icon={showKeys[provider] ? <FiEyeOff /> : <FiEye />}
                          onClick={() => setShowKeys({ ...showKeys, [provider]: !showKeys[provider] })}
                        />
                      </InputRightElement>
                    </InputGroup>
                    <Button size="sm" leftIcon={<FiSave />} onClick={() => handleSaveApiKey(provider)}>
                      Save
                    </Button>
                  </HStack>
                </FormControl>
              ))}

              <Divider />

              {/* Custom Models - Only for paid providers */}
              <FormControl>
                <FormLabel>Custom Models (Paid Providers)</FormLabel>
                <Text fontSize="xs" color="gray.600" mb={2}>
                  Add custom model names for OpenAI, Anthropic, Google, Mistral. Ollama models load automatically.
                </Text>
                <HStack mb={2}>
                  <Select
                    size="sm"
                    value={selectedProvider}
                    onChange={(e) => setSelectedProvider(e.target.value)}
                  >
                    <option value="openai">OpenAI</option>
                    <option value="anthropic">Anthropic</option>
                    <option value="google">Google</option>
                    <option value="mistral">Mistral</option>
                  </Select>
                  <Input
                    size="sm"
                    placeholder="e.g., gpt-4o-2024-11-20"
                    value={newModel}
                    onChange={(e) => setNewModel(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleAddModel()}
                  />
                  <IconButton size="sm" icon={<FiPlus />} colorScheme="green" onClick={handleAddModel} />
                </HStack>
                {customModels[selectedProvider]?.length > 0 && (
                  <VStack align="stretch" spacing={1} maxH="150px" overflowY="auto">
                    {customModels[selectedProvider].map((model, i) => (
                      <HStack key={i} justify="space-between" fontSize="sm" p={1} bg="gray.50" borderRadius="sm">
                        <Code fontSize="xs">{model}</Code>
                        <IconButton
                          size="xs"
                          icon={<FiTrash2 />}
                          variant="ghost"
                          colorScheme="red"
                          onClick={() => handleRemoveModel(selectedProvider, model)}
                        />
                      </HStack>
                    ))}
                  </VStack>
                )}
              </FormControl>
            </VStack>
          </CardBody>
        </Card>

        {/* System Settings */}
        <Card bg={cardBg} borderWidth="1px" borderColor={borderColor}>
          <CardHeader>
            <Heading size="md">System Settings</Heading>
          </CardHeader>
          <Divider />
          <CardBody>
            <VStack spacing={4} align="stretch">
              <FormControl>
                <FormLabel>Data Directory</FormLabel>
                <Input
                  size="sm"
                  value={systemSettings.dataDirectory}
                  onChange={(e) => setSystemSettings({ ...systemSettings, dataDirectory: e.target.value })}
                />
              </FormControl>

              <FormControl>
                <FormLabel>Max Concurrent Jobs</FormLabel>
                <NumberInput
                  size="sm"
                  value={systemSettings.maxConcurrentJobs}
                  onChange={(value) => setSystemSettings({ ...systemSettings, maxConcurrentJobs: parseInt(value) })}
                  min={1}
                  max={10}
                >
                  <NumberInputField />
                  <NumberInputStepper>
                    <NumberIncrementStepper />
                    <NumberDecrementStepper />
                  </NumberInputStepper>
                </NumberInput>
              </FormControl>

              <FormControl display="flex" alignItems="center">
                <Switch
                  isChecked={systemSettings.enableCaching}
                  onChange={(e) => setSystemSettings({ ...systemSettings, enableCaching: e.target.checked })}
                  mr={3}
                />
                <FormLabel mb="0">Enable Caching</FormLabel>
              </FormControl>

              {systemSettings.enableCaching && (
                <FormControl>
                  <FormLabel>Cache TTL (hours)</FormLabel>
                  <NumberInput
                    size="sm"
                    value={systemSettings.cacheTTL}
                    onChange={(value) => setSystemSettings({ ...systemSettings, cacheTTL: parseInt(value) })}
                    min={1}
                    max={168}
                  >
                    <NumberInputField />
                    <NumberInputStepper>
                      <NumberIncrementStepper />
                      <NumberDecrementStepper />
                    </NumberInputStepper>
                  </NumberInput>
                </FormControl>
              )}

              <FormControl>
                <FormLabel>Log Level</FormLabel>
                <Select
                  size="sm"
                  value={systemSettings.logLevel}
                  onChange={(e) => setSystemSettings({ ...systemSettings, logLevel: e.target.value })}
                >
                  <option value="debug">Debug</option>
                  <option value="info">Info</option>
                  <option value="warning">Warning</option>
                  <option value="error">Error</option>
                </Select>
              </FormControl>

              <Button colorScheme="blue" onClick={saveSystemSettings} size="sm" alignSelf="flex-start">
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
