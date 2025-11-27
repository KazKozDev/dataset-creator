import React, { useState, useEffect } from 'react';
import {
  Box,
  Button,
  FormControl,
  FormLabel,
  Heading,
  Select,
  Text,
  useToast,
  Progress,
  Badge,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  NumberIncrementStepper,
  NumberDecrementStepper,
  Switch,
  VStack,
  HStack,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Card,
  CardBody,
  useColorModeValue,
  Tooltip,
  SimpleGrid,
  Divider,
  Alert,
  AlertIcon
} from '@chakra-ui/react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { Link, useLocation } from 'react-router-dom';
import { getDatasets, getProviders, getProviderModels, startQualityControl, getQualityStatus } from '../services/api';

function QualityControl() {
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [batchSize, setBatchSize] = useState(10);
  const [threshold, setThreshold] = useState(7.0);
  const [autoFix, setAutoFix] = useState(false);
  const [autoRemove, setAutoRemove] = useState(false);
  const [activeJobId, setActiveJobId] = useState(null);
  const [provider, setProvider] = useState('ollama');
  const [model, setModel] = useState('');

  const toast = useToast();
  const location = useLocation();
  const cardBg = useColorModeValue('white', 'gray.700');
  const borderColor = useColorModeValue('gray.200', 'gray.600');

  // Fetch datasets
  const {
    data: datasets,
    isLoading: isLoadingDatasets,
    isError: isDatasetsError,
    error: datasetsError,
    refetch: refetchDatasets
  } = useQuery({
    queryKey: ['datasets'],
    queryFn: getDatasets,
    retry: 1
  });

  // Fetch providers
  const {
    data: providersData,
    isLoading: isLoadingProviders,
    isError: isProvidersError,
    error: providersError,
    refetch: refetchProviders
  } = useQuery({
    queryKey: ['providers'],
    queryFn: getProviders,
    retry: 1
  });

  // Fetch models for selected provider
  const { data: modelsData, isLoading: isLoadingModels } = useQuery({
    queryKey: ['providerModels', provider],
    queryFn: () => getProviderModels(provider),
    enabled: !!provider
  });

  // Fetch job status
  const { data: jobStatus } = useQuery({
    queryKey: ['quality', 'status', activeJobId],
    queryFn: () => getQualityStatus(activeJobId),
    enabled: !!activeJobId,
    refetchInterval: (data) => {
      if (data?.status === 'completed' || data?.status === 'failed' || data?.status === 'cancelled') {
        return false;
      }
      return 1000;
    }
  });

  // Start quality control mutation
  const startQualityControlMutation = useMutation({
    mutationFn: (params) => startQualityControl(params),
    onSuccess: (data) => {
      setActiveJobId(data.job_id);
      toast({
        title: 'Quality control started',
        description: `Job ID: ${data.job_id}`,
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
    },
    onError: (error) => {
      toast({
        title: 'Failed to start quality control',
        description: error.response?.data?.detail || 'Unknown error',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    },
  });

  // Handle form submission
  const handleStartQualityControl = () => {
    if (!selectedDataset) {
      toast({
        title: 'No dataset selected',
        description: 'Please select a dataset to analyze',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    if (!model) {
      toast({
        title: 'No model selected',
        description: 'Please select a model to use',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    startQualityControlMutation.mutate({
      dataset_id: selectedDataset,
      batch_size: batchSize,
      threshold: threshold,
      auto_fix: autoFix,
      auto_remove: autoRemove,
      provider: provider,
      model: model,
    });
  };

  // Helper function to safely render any value
  const safeRender = (value) => {
    if (value === null || value === undefined) return '';
    if (typeof value === 'object') return JSON.stringify(value);
    return String(value);
  };

  // Calculate progress percentage
  const calculateProgress = () => {
    if (!jobStatus) return 0;
    const processed = Number(jobStatus.examples_processed) || 0;
    const total = Number(jobStatus.total_examples) || 1;
    return (processed / total) * 100;
  };

  // Get dataset name by ID
  const getDatasetName = (id) => {
    if (typeof id === 'object') return JSON.stringify(id);
    const dataset = datasets?.datasets?.find(d => d.id === id);
    if (!dataset) return String(id || '');
    return safeRender(dataset.name);
  };

  // Parse dataset ID from URL if provided
  useEffect(() => {
    const params = new URLSearchParams(location.search);
    const datasetId = params.get('dataset');
    if (datasetId) {
      setSelectedDataset(parseInt(datasetId));
    }
  }, [location]);

  // Show loading state
  if (isLoadingDatasets || isLoadingProviders) {
    return (
      <Box>
        <Heading size="md" mb={6}>Quality Control</Heading>
        <Card bg={cardBg} borderWidth="1px" borderColor={borderColor} p={6}>
          <CardBody>
            <VStack spacing={4}>
              <Progress size="xs" isIndeterminate width="100%" />
              <Text>Loading...</Text>
            </VStack>
          </CardBody>
        </Card>
      </Box>
    );
  }

  // Show error state
  if (isDatasetsError || isProvidersError) {
    return (
      <Box>
        <Heading size="md" mb={6}>Quality Control</Heading>
        <Card bg={cardBg} borderWidth="1px" borderColor={borderColor} p={6}>
          <CardBody>
            <Alert status="error" flexDirection="column" alignItems="center" justifyContent="center" textAlign="center" borderRadius="md">
              <AlertIcon boxSize="40px" mr={0} />
              <Heading size="md" mt={4} mb={1}>
                Failed to load data
              </Heading>
              <Text mb={4}>
                {datasetsError?.message || providersError?.message || 'Could not connect to the server.'}
              </Text>
              <Button colorScheme="red" onClick={() => { refetchDatasets(); refetchProviders(); }}>
                Retry
              </Button>
            </Alert>
          </CardBody>
        </Card>
      </Box>
    );
  }

  return (
    <Box>
      <Heading size="md" mb={6}>Quality Control</Heading>

      <Card bg={cardBg} borderWidth="1px" borderColor={borderColor} p={6}>
        <CardBody>
          <VStack spacing={6} align="stretch">
            <SimpleGrid columns={{ base: 1, md: 2 }} spacing={6}>
              <VStack spacing={6} align="stretch">
                <FormControl>
                  <FormLabel>Dataset</FormLabel>
                  <Tooltip
                    label={selectedDataset ? `${getDatasetName(selectedDataset)} (${safeRender(datasets?.datasets?.find(d => d.id === selectedDataset)?.example_count)} examples)` : "Select a dataset"}
                    placement="top"
                    isDisabled={!selectedDataset}
                  >
                    <Select
                      value={selectedDataset || ''}
                      onChange={(e) => setSelectedDataset(e.target.value ? parseInt(e.target.value) : null)}
                      placeholder="Select a dataset"
                      maxWidth="300px"
                    >
                      {datasets?.datasets?.map((dataset) => (
                        <option key={dataset.id} value={dataset.id}>
                          {safeRender(dataset.name)} ({safeRender(dataset.example_count)} examples)
                        </option>
                      ))}
                    </Select>
                  </Tooltip>
                </FormControl>

                <FormControl>
                  <FormLabel>Provider</FormLabel>
                  <Select
                    value={provider}
                    onChange={(e) => {
                      setProvider(e.target.value);
                      setModel('');
                    }}
                    maxWidth="300px"
                  >
                    {providersData?.providers?.map((p) => (
                      <option key={p.id} value={p.id} disabled={!p.available}>
                        {safeRender(p.name)} {!p.available && '(API key required)'}
                      </option>
                    ))}
                  </Select>
                </FormControl>

                <FormControl>
                  <FormLabel>Model</FormLabel>
                  <Tooltip
                    label={model || "Select a model"}
                    placement="top"
                    isDisabled={!model}
                  >
                    <Select
                      value={model}
                      onChange={(e) => setModel(e.target.value)}
                      placeholder="Select a model"
                      isDisabled={isLoadingModels}
                      maxWidth="300px"
                    >
                      {isLoadingModels ? (
                        <option>Loading models...</option>
                      ) : Array.isArray(modelsData) && modelsData.length > 0 ? (
                        modelsData.map((m) => (
                          <option key={m.id} value={m.id}>
                            {safeRender(m.name)}
                          </option>
                        ))
                      ) : (
                        <option value="">No models available</option>
                      )}
                    </Select>
                  </Tooltip>
                </FormControl>

                <Box mt={8}>
                  <FormControl display="flex" alignItems="center">
                    <Switch
                      isChecked={autoFix}
                      onChange={(e) => setAutoFix(e.target.checked)}
                      mr={3}
                    />
                    <FormLabel mb="0">Auto-Fix Low Quality Examples</FormLabel>
                  </FormControl>

                  <FormControl display="flex" alignItems="center" mt={4}>
                    <Switch
                      isChecked={autoRemove}
                      onChange={(e) => setAutoRemove(e.target.checked)}
                      mr={3}
                    />
                    <FormLabel mb="0">Auto-Remove Very Low Quality Examples</FormLabel>
                  </FormControl>
                </Box>
              </VStack>

              <VStack spacing={6} align="stretch">
                <FormControl>
                  <FormLabel>Batch Size</FormLabel>
                  <NumberInput
                    value={batchSize}
                    onChange={(value) => setBatchSize(Number(value))}
                    min={1}
                    max={100}
                    maxWidth="300px"
                  >
                    <NumberInputField />
                    <NumberInputStepper>
                      <NumberIncrementStepper />
                      <NumberDecrementStepper />
                    </NumberInputStepper>
                  </NumberInput>
                </FormControl>

                <FormControl>
                  <FormLabel>Quality Threshold (0-10)</FormLabel>
                  <NumberInput
                    value={threshold}
                    onChange={(value) => setThreshold(Number(value))}
                    min={0}
                    max={10}
                    step={0.1}
                    maxWidth="300px"
                  >
                    <NumberInputField />
                    <NumberInputStepper>
                      <NumberIncrementStepper />
                      <NumberDecrementStepper />
                    </NumberInputStepper>
                  </NumberInput>
                </FormControl>
              </VStack>
            </SimpleGrid>

            <Divider my={4} />

            <Button
              colorScheme="blue"
              onClick={handleStartQualityControl}
              isDisabled={!selectedDataset || activeJobId}
              isLoading={startQualityControlMutation.isPending}
              width="200px"
              alignSelf="flex-start"
            >
              Start Quality Control
            </Button>

            {jobStatus && (
              <Box mt={6}>
                <Text fontSize="lg" mb={4}>
                  Analyzing {getDatasetName(jobStatus.dataset_id)}
                </Text>

                <HStack spacing={4} mb={4}>
                  <Badge colorScheme={jobStatus.status === 'completed' ? 'green' : jobStatus.status === 'failed' ? 'red' : 'blue'}>
                    {safeRender(jobStatus.status)}
                  </Badge>
                  <Text>
                    {safeRender(jobStatus.examples_processed)} / {safeRender(jobStatus.total_examples)} examples processed
                  </Text>
                </HStack>

                <Progress
                  value={calculateProgress()}
                  size="lg"
                  colorScheme="blue"
                  mb={4}
                />

                {jobStatus.status === 'completed' && (
                  <VStack spacing={4} align="stretch">
                    <Text>Quality Analysis Results:</Text>
                    <Table variant="simple">
                      <Thead>
                        <Tr>
                          <Th>Metric</Th>
                          <Th isNumeric>Value</Th>
                        </Tr>
                      </Thead>
                      <Tbody>
                        <Tr>
                          <Td>Average Quality Score</Td>
                          <Td isNumeric>{typeof jobStatus.average_score === 'number' ? jobStatus.average_score.toFixed(2) : safeRender(jobStatus.average_score)}</Td>
                        </Tr>
                        <Tr>
                          <Td>Examples Below Threshold</Td>
                          <Td isNumeric>{safeRender(jobStatus.below_threshold_count)}</Td>
                        </Tr>
                        <Tr>
                          <Td>Examples Fixed</Td>
                          <Td isNumeric>{safeRender(jobStatus.fixed_count)}</Td>
                        </Tr>
                        <Tr>
                          <Td>Examples Removed</Td>
                          <Td isNumeric>{safeRender(jobStatus.removed_count)}</Td>
                        </Tr>
                      </Tbody>
                    </Table>

                    <Button
                      as={Link}
                      to={`/datasets/${jobStatus.dataset_id}`}
                      colorScheme="blue"
                      width="200px"
                      alignSelf="flex-start"
                    >
                      View Dataset
                    </Button>
                  </VStack>
                )}

                {jobStatus.status === 'failed' && (
                  <Text color="red.500">
                    Error: {(() => {
                      if (jobStatus.error) {
                        return typeof jobStatus.error === 'object'
                          ? JSON.stringify(jobStatus.error)
                          : String(jobStatus.error);
                      }
                      if (Array.isArray(jobStatus.errors)) {
                        return jobStatus.errors
                          .map(e => typeof e === 'object' ? JSON.stringify(e) : String(e))
                          .join(', ');
                      }
                      return 'Unknown error occurred';
                    })()}
                  </Text>
                )}
              </Box>
            )}
          </VStack>
        </CardBody>
      </Card>
    </Box>
  );
}

export default QualityControl;
