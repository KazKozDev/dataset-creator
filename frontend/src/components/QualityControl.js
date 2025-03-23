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
  Divider
} from '@chakra-ui/react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { Link, useLocation } from 'react-router-dom';
import { getDatasets, getModels, startQualityControl, getQualityStatus } from '../services/api';

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
  const { data: datasets } = useQuery({
    queryKey: ['datasets'],
    queryFn: getDatasets
  });

  // Fetch models for selected provider
  const { data: modelsData, isLoading: isLoadingModels } = useQuery({
    queryKey: ['models', provider],
    queryFn: () => getModels(provider),
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

  // Calculate progress percentage
  const calculateProgress = () => {
    if (!jobStatus) return 0;
    return (jobStatus.examples_processed / jobStatus.total_examples) * 100;
  };

  // Get dataset name by ID
  const getDatasetName = (id) => {
    const dataset = datasets?.datasets?.find(d => d.id === id);
    return dataset ? dataset.name : id;
  };

  // Parse dataset ID from URL if provided
  useEffect(() => {
    const params = new URLSearchParams(location.search);
    const datasetId = params.get('dataset');
    if (datasetId) {
      setSelectedDataset(parseInt(datasetId));
    }
  }, [location]);

  return (
    <Box>
      <Heading size="lg" mb={6}>Quality Control</Heading>

      <Card bg={cardBg} borderWidth="1px" borderColor={borderColor} p={6}>
        <CardBody>
          <VStack spacing={6} align="stretch">
            <SimpleGrid columns={{ base: 1, md: 2 }} spacing={6}>
              <VStack spacing={6} align="stretch">
                <FormControl>
                  <FormLabel>Dataset</FormLabel>
                  <Tooltip 
                    label={selectedDataset ? `${getDatasetName(selectedDataset)} (${datasets?.datasets?.find(d => d.id === selectedDataset)?.example_count} examples)` : "Select a dataset"} 
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
                          {dataset.name} ({dataset.example_count} examples)
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
                    <option value="ollama">Ollama</option>
                    <option value="openai">OpenAI</option>
                    <option value="anthropic">Anthropic</option>
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
                      ) : provider === 'openai' ? (
                        <>
                          <option value="gpt-4o">GPT-4o</option>
                          <option value="03-mini-high">03-mini-high</option>
                        </>
                      ) : provider === 'anthropic' ? (
                        <>
                          <option value="claude-3-7-sonnet-20250219">Claude 3.7 Sonnet</option>
                        </>
                      ) : modelsData?.models?.length > 0 ? (
                        modelsData.models.map((model) => (
                          <option key={model} value={model}>
                            {model}
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
                    {jobStatus.status}
                  </Badge>
                  <Text>
                    {jobStatus.examples_processed} / {jobStatus.total_examples} examples processed
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
                          <Td isNumeric>{jobStatus.average_score?.toFixed(2)}</Td>
                        </Tr>
                        <Tr>
                          <Td>Examples Below Threshold</Td>
                          <Td isNumeric>{jobStatus.below_threshold_count}</Td>
                        </Tr>
                        <Tr>
                          <Td>Examples Fixed</Td>
                          <Td isNumeric>{jobStatus.fixed_count}</Td>
                        </Tr>
                        <Tr>
                          <Td>Examples Removed</Td>
                          <Td isNumeric>{jobStatus.removed_count}</Td>
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
                    Error: {jobStatus.error || 'Unknown error occurred'}
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
