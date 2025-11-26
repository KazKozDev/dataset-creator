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
  CardHeader,
  useColorModeValue,
  Tooltip,
  SimpleGrid,
  Divider,
  Progress,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  Spinner,
  Accordion,
  AccordionItem,
  AccordionButton,
  AccordionPanel,
  AccordionIcon,
} from '@chakra-ui/react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { Link, useLocation } from 'react-router-dom';
import { getDatasets, getProviders, getProviderModels, startQualityControl, getQualityStatus } from '../services/api';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Legend,
} from 'recharts';

function Quality() {
  // Common state
  const [selectedDataset, setSelectedDataset] = useState(null);
  const toast = useToast();
  const location = useLocation();
  const cardBg = useColorModeValue('white', 'gray.700');
  const borderColor = useColorModeValue('gray.200', 'gray.600');
  const warningBg = useColorModeValue('orange.50', 'orange.900');

  // AI Quality Check state
  const [batchSize, setBatchSize] = useState(10);
  const [threshold, setThreshold] = useState(7.0);
  const [autoFix, setAutoFix] = useState(false);
  const [autoRemove, setAutoRemove] = useState(false);
  const [activeJobId, setActiveJobId] = useState(null);
  const [provider, setProvider] = useState('ollama');
  const [model, setModel] = useState('');

  // Safety Metrics state
  const [analysisResults, setAnalysisResults] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Dataset Analytics state
  const [analyticsResults, setAnalyticsResults] = useState(null);
  const [isAnalyzingAnalytics, setIsAnalyzingAnalytics] = useState(false);

  // Fetch datasets
  const { data: datasets, isLoading: isDatasetsLoading } = useQuery({
    queryKey: ['datasets'],
    queryFn: getDatasets
  });

  // Fetch providers
  const { data: providersData } = useQuery({
    queryKey: ['providers'],
    queryFn: getProviders,
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
        title: 'AI Quality Check started',
        description: `Job ID: ${data.job_id}`,
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
    },
    onError: (error) => {
      toast({
        title: 'Failed to start quality check',
        description: error.response?.data?.detail || 'Unknown error',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    },
  });

  // Handle AI quality control
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

  // Full safety analysis
  const analyzeFullQuality = async () => {
    const response = await fetch(
      `http://localhost:8000/api/quality/analyze/full?dataset_id=${selectedDataset}`,
      { method: 'POST' }
    );
    if (!response.ok) throw new Error('Failed to analyze quality');
    return response.json();
  };

  // Anonymize PII
  const anonymizePII = async () => {
    const response = await fetch(
      `http://localhost:8000/api/quality/anonymize/pii?dataset_id=${selectedDataset}`,
      { method: 'POST' }
    );
    if (!response.ok) throw new Error('Failed to anonymize PII');
    return response.json();
  };

  // Handle safety analysis
  const handleSafetyAnalysis = async () => {
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

    setIsAnalyzing(true);
    try {
      const results = await analyzeFullQuality();
      setAnalysisResults(results);
      toast({
        title: 'Safety analysis complete',
        description: 'Quality metrics have been calculated',
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
    } catch (error) {
      toast({
        title: 'Analysis failed',
        description: error.message,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Handle anonymization
  const handleAnonymization = async () => {
    if (!selectedDataset) {
      toast({
        title: 'No dataset selected',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
      return;
    }

    try {
      const result = await anonymizePII();
      toast({
        title: 'PII Anonymized',
        description: result.message,
        status: 'success',
        duration: 5000,
        isClosable: true,
      });
    } catch (error) {
      toast({
        title: 'Anonymization failed',
        description: error.message,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  };

  // Dataset Analytics API call
  const runDatasetAnalytics = async () => {
    const response = await fetch(
      `http://localhost:8000/api/quality/analytics?dataset_id=${selectedDataset}`,
      { method: 'POST' }
    );
    if (!response.ok) throw new Error('Failed to run analytics');
    return response.json();
  };

  // Handle dataset analytics
  const handleDatasetAnalytics = async () => {
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

    setIsAnalyzingAnalytics(true);
    try {
      const results = await runDatasetAnalytics();
      setAnalyticsResults(results);
      toast({
        title: 'Analytics complete',
        description: 'Dataset metrics have been calculated',
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
    } catch (error) {
      toast({
        title: 'Analytics failed',
        description: error.message,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsAnalyzingAnalytics(false);
    }
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
      <Heading size="md" mb={6}>Quality Control & Safety</Heading>

      {/* Dataset Selection */}
      <Card bg={cardBg} borderWidth="1px" borderColor={borderColor} mb={6}>
        <CardHeader>
          <Heading size="sm">Dataset Selection</Heading>
        </CardHeader>
        <CardBody>
          <FormControl maxW="500px">
            <FormLabel>Select Dataset</FormLabel>
            <Tooltip
              label={selectedDataset ? `${getDatasetName(selectedDataset)} (${datasets?.datasets?.find(d => d.id === selectedDataset)?.example_count} examples)` : "Select a dataset"}
              placement="top"
              isDisabled={!selectedDataset}
            >
              <Select
                value={selectedDataset || ''}
                onChange={(e) => setSelectedDataset(e.target.value ? parseInt(e.target.value) : null)}
                placeholder="Select a dataset"
                isDisabled={isDatasetsLoading}
              >
                {datasets?.datasets?.map((dataset) => (
                  <option key={dataset.id} value={dataset.id}>
                    {dataset.name} ({dataset.example_count} examples)
                  </option>
                ))}
              </Select>
            </Tooltip>
          </FormControl>
        </CardBody>
      </Card>

      {/* Tabs for different quality checks */}
      <Card bg={cardBg} borderWidth="1px" borderColor={borderColor}>
        <CardBody>
          <Tabs variant="line" colorScheme="blue">
            <TabList mb={4}>
              <Tab>Dataset Analytics</Tab>
              <Tab>Safety & Compliance</Tab>
            </TabList>

            <TabPanels>
              {/* Dataset Analytics Tab - Now First */}
              <TabPanel>
                <VStack spacing={6} align="stretch">
                  <Alert status="info">
                    <AlertIcon />
                    <Box>
                      <AlertTitle>Dataset Analytics</AlertTitle>
                      <AlertDescription>
                        Comprehensive analysis of dataset diversity, text statistics, and semantic clustering. 
                        No LLM required - runs locally using statistical methods.
                      </AlertDescription>
                    </Box>
                  </Alert>

                  <HStack spacing={4}>
                    <Button
                      colorScheme="blue"
                      onClick={handleDatasetAnalytics}
                      isDisabled={!selectedDataset}
                      isLoading={isAnalyzingAnalytics}
                    >
                      Run Dataset Analytics
                    </Button>
                  </HStack>

                  {isAnalyzingAnalytics && (
                    <Card bg={cardBg} borderWidth="1px" borderColor={borderColor}>
                      <CardBody>
                        <VStack spacing={4}>
                          <Spinner size="xl" color="blue.500" />
                          <Text>Analyzing dataset...</Text>
                          <Text fontSize="sm" color="gray.500">
                            Calculating diversity metrics, text statistics, and semantic clusters
                          </Text>
                        </VStack>
                      </CardBody>
                    </Card>
                  )}

                  {analyticsResults && (
                    <>
                      {/* Overall Scores */}
                      <Card bg={cardBg} borderWidth="1px" borderColor={borderColor}>
                        <CardHeader>
                          <Heading size="sm">Overall Scores</Heading>
                        </CardHeader>
                        <CardBody>
                          <SimpleGrid columns={{ base: 1, md: 4 }} spacing={6}>
                            <Stat>
                              <StatLabel>Total Examples</StatLabel>
                              <StatNumber>{analyticsResults.total_examples}</StatNumber>
                            </Stat>
                            <Stat>
                              <StatLabel>Diversity Score</StatLabel>
                              <StatNumber color={analyticsResults.overall_diversity_score > 0.5 ? 'green.500' : 'orange.500'}>
                                {(analyticsResults.overall_diversity_score * 100).toFixed(1)}%
                              </StatNumber>
                              <StatHelpText>Higher is better</StatHelpText>
                            </Stat>
                            <Stat>
                              <StatLabel>Quality Score</StatLabel>
                              <StatNumber color={analyticsResults.overall_quality_score > 0.5 ? 'green.500' : 'orange.500'}>
                                {(analyticsResults.overall_quality_score * 100).toFixed(1)}%
                              </StatNumber>
                              <StatHelpText>Higher is better</StatHelpText>
                            </Stat>
                            <Stat>
                              <StatLabel>Vocabulary Size</StatLabel>
                              <StatNumber>{analyticsResults.diversity?.vocabulary_size?.toLocaleString()}</StatNumber>
                              <StatHelpText>Unique words</StatHelpText>
                            </Stat>
                          </SimpleGrid>

                          {/* Warnings */}
                          {analyticsResults.warnings && analyticsResults.warnings.length > 0 && (
                            <VStack spacing={2} mt={4} align="stretch">
                              {analyticsResults.warnings.map((warning, idx) => (
                                <Alert key={idx} status="warning" size="sm">
                                  <AlertIcon />
                                  {warning}
                                </Alert>
                              ))}
                            </VStack>
                          )}

                          {/* Deep AI Review Section */}
                          {(analyticsResults.overall_quality_score < 0.6 || 
                            analyticsResults.text_stats?.very_short_responses > 0 ||
                            analyticsResults.text_stats?.empty_responses > 0) && (
                            <Box mt={6} p={4} borderWidth="1px" borderRadius="md" borderColor="orange.300" bg={warningBg}>
                              <HStack spacing={3} mb={3}>
                                <AlertIcon color="orange.500" />
                                <Text fontWeight="bold">Issues Detected - Deep AI Review Available</Text>
                              </HStack>
                              <Text fontSize="sm" mb={4}>
                                Found {(analyticsResults.text_stats?.very_short_responses || 0) + (analyticsResults.text_stats?.empty_responses || 0)} potentially problematic examples. 
                                Use AI to review and optionally fix them.
                              </Text>
                              
                              <HStack spacing={4} wrap="wrap">
                                <FormControl maxW="200px">
                                  <FormLabel fontSize="sm">Provider</FormLabel>
                                  <Select
                                    size="sm"
                                    value={provider}
                                    onChange={(e) => {
                                      setProvider(e.target.value);
                                      setModel('');
                                    }}
                                  >
                                    {providersData?.providers?.map((p) => (
                                      <option key={p.id} value={p.id} disabled={!p.available}>
                                        {p.name}
                                      </option>
                                    ))}
                                  </Select>
                                </FormControl>
                                
                                <FormControl maxW="200px">
                                  <FormLabel fontSize="sm">Model</FormLabel>
                                  <Select
                                    size="sm"
                                    value={model}
                                    onChange={(e) => setModel(e.target.value)}
                                    placeholder="Select model"
                                    isDisabled={isLoadingModels}
                                  >
                                    {Array.isArray(modelsData) && modelsData.map((m) => (
                                      <option key={m.id} value={m.id}>{m.name}</option>
                                    ))}
                                  </Select>
                                </FormControl>

                                <FormControl display="flex" alignItems="center" maxW="180px" mt={6}>
                                  <Switch
                                    size="sm"
                                    isChecked={autoFix}
                                    onChange={(e) => setAutoFix(e.target.checked)}
                                    mr={2}
                                  />
                                  <FormLabel fontSize="sm" mb="0">Auto-Fix</FormLabel>
                                </FormControl>

                                <Button
                                  colorScheme="orange"
                                  size="sm"
                                  onClick={handleStartQualityControl}
                                  isDisabled={!model || activeJobId}
                                  isLoading={startQualityControlMutation.isPending}
                                  mt={6}
                                >
                                  🔍 Deep AI Review
                                </Button>
                              </HStack>

                              {jobStatus && (
                                <Box mt={4}>
                                  <HStack spacing={4} mb={2}>
                                    <Badge colorScheme={jobStatus.status === 'completed' ? 'green' : jobStatus.status === 'failed' ? 'red' : 'blue'}>
                                      {jobStatus.status}
                                    </Badge>
                                    <Text fontSize="sm">
                                      {jobStatus.examples_processed} / {jobStatus.total_examples} processed
                                    </Text>
                                  </HStack>
                                  <Progress value={calculateProgress()} size="sm" colorScheme="blue" />
                                  
                                  {jobStatus.status === 'completed' && (
                                    <SimpleGrid columns={4} spacing={4} mt={3}>
                                      <Stat size="sm">
                                        <StatLabel fontSize="xs">Avg Score</StatLabel>
                                        <StatNumber fontSize="md">{jobStatus.average_score?.toFixed(1)}</StatNumber>
                                      </Stat>
                                      <Stat size="sm">
                                        <StatLabel fontSize="xs">Below Threshold</StatLabel>
                                        <StatNumber fontSize="md">{jobStatus.below_threshold_count}</StatNumber>
                                      </Stat>
                                      <Stat size="sm">
                                        <StatLabel fontSize="xs">Fixed</StatLabel>
                                        <StatNumber fontSize="md" color="green.500">{jobStatus.fixed_count}</StatNumber>
                                      </Stat>
                                      <Stat size="sm">
                                        <StatLabel fontSize="xs">Removed</StatLabel>
                                        <StatNumber fontSize="md" color="red.500">{jobStatus.removed_count}</StatNumber>
                                      </Stat>
                                    </SimpleGrid>
                                  )}
                                </Box>
                              )}
                            </Box>
                          )}
                        </CardBody>
                      </Card>

                      {/* Detailed Metrics Tabs */}
                      <Card bg={cardBg} borderWidth="1px" borderColor={borderColor}>
                        <CardBody>
                          <Tabs variant="line" colorScheme="blue">
                            <TabList mb={4}>
                              <Tab>Diversity</Tab>
                              <Tab>Text Stats</Tab>
                              <Tab>Semantic Clusters</Tab>
                              <Tab>Recommendations</Tab>
                            </TabList>

                            <TabPanels>
                              {/* Diversity Tab */}
                              <TabPanel>
                                <VStack spacing={4} align="stretch">
                                  <Text fontWeight="bold">Diversity Metrics</Text>
                                  
                                  <SimpleGrid columns={{ base: 1, md: 2 }} spacing={6}>
                                    <Box>
                                      <Text fontWeight="semibold" mb={2}>Distinct-n Scores</Text>
                                      <Text fontSize="sm" color="gray.500" mb={2}>
                                        Ratio of unique n-grams. Higher = more diverse vocabulary.
                                      </Text>
                                      <Table variant="simple" size="sm">
                                        <Tbody>
                                          <Tr>
                                            <Td>Distinct-1 (unigrams)</Td>
                                            <Td isNumeric>
                                              <Badge colorScheme={analyticsResults.diversity?.distinct_1 > 0.3 ? 'green' : 'orange'}>
                                                {(analyticsResults.diversity?.distinct_1 * 100).toFixed(1)}%
                                              </Badge>
                                            </Td>
                                          </Tr>
                                          <Tr>
                                            <Td>Distinct-2 (bigrams)</Td>
                                            <Td isNumeric>
                                              <Badge colorScheme={analyticsResults.diversity?.distinct_2 > 0.5 ? 'green' : 'orange'}>
                                                {(analyticsResults.diversity?.distinct_2 * 100).toFixed(1)}%
                                              </Badge>
                                            </Td>
                                          </Tr>
                                          <Tr>
                                            <Td>Distinct-3 (trigrams)</Td>
                                            <Td isNumeric>
                                              <Badge colorScheme={analyticsResults.diversity?.distinct_3 > 0.6 ? 'green' : 'orange'}>
                                                {(analyticsResults.diversity?.distinct_3 * 100).toFixed(1)}%
                                              </Badge>
                                            </Td>
                                          </Tr>
                                        </Tbody>
                                      </Table>
                                    </Box>

                                    <Box>
                                      <Text fontWeight="semibold" mb={2}>Self-BLEU Scores</Text>
                                      <Text fontSize="sm" color="gray.500" mb={2}>
                                        Similarity between examples. Lower = more diverse content.
                                      </Text>
                                      <Table variant="simple" size="sm">
                                        <Tbody>
                                          <Tr>
                                            <Td>Self-BLEU-1</Td>
                                            <Td isNumeric>
                                              <Badge colorScheme={analyticsResults.diversity?.self_bleu_1 < 0.5 ? 'green' : 'orange'}>
                                                {(analyticsResults.diversity?.self_bleu_1 * 100).toFixed(1)}%
                                              </Badge>
                                            </Td>
                                          </Tr>
                                          <Tr>
                                            <Td>Self-BLEU-2</Td>
                                            <Td isNumeric>
                                              <Badge colorScheme={analyticsResults.diversity?.self_bleu_2 < 0.4 ? 'green' : 'orange'}>
                                                {(analyticsResults.diversity?.self_bleu_2 * 100).toFixed(1)}%
                                              </Badge>
                                            </Td>
                                          </Tr>
                                          <Tr>
                                            <Td>Self-BLEU-3</Td>
                                            <Td isNumeric>
                                              <Badge colorScheme={analyticsResults.diversity?.self_bleu_3 < 0.3 ? 'green' : 'orange'}>
                                                {(analyticsResults.diversity?.self_bleu_3 * 100).toFixed(1)}%
                                              </Badge>
                                            </Td>
                                          </Tr>
                                          <Tr>
                                            <Td fontWeight="bold">Average</Td>
                                            <Td isNumeric>
                                              <Badge colorScheme={analyticsResults.diversity?.self_bleu_avg < 0.4 ? 'green' : 'orange'}>
                                                {(analyticsResults.diversity?.self_bleu_avg * 100).toFixed(1)}%
                                              </Badge>
                                            </Td>
                                          </Tr>
                                        </Tbody>
                                      </Table>
                                    </Box>
                                  </SimpleGrid>

                                  <Divider />

                                  {/* Diversity Radar Chart */}
                                  <Text fontWeight="semibold" mb={2}>Diversity Overview</Text>
                                  <Box height="300px">
                                    <ResponsiveContainer width="100%" height="100%">
                                      <RadarChart
                                        data={[
                                          { metric: 'Distinct-1', value: (analyticsResults.diversity?.distinct_1 || 0) * 100, fullMark: 100 },
                                          { metric: 'Distinct-2', value: (analyticsResults.diversity?.distinct_2 || 0) * 100, fullMark: 100 },
                                          { metric: 'Distinct-3', value: (analyticsResults.diversity?.distinct_3 || 0) * 100, fullMark: 100 },
                                          { metric: 'Low Self-BLEU', value: (1 - (analyticsResults.diversity?.self_bleu_avg || 0)) * 100, fullMark: 100 },
                                          { metric: 'Vocab Richness', value: Math.min((analyticsResults.diversity?.vocabulary_size || 0) / 100, 100), fullMark: 100 },
                                        ]}
                                      >
                                        <PolarGrid />
                                        <PolarAngleAxis dataKey="metric" tick={{ fontSize: 12 }} />
                                        <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 10 }} />
                                        <Radar name="Diversity" dataKey="value" stroke="#3182CE" fill="#3182CE" fillOpacity={0.5} />
                                      </RadarChart>
                                    </ResponsiveContainer>
                                  </Box>

                                  <Divider />

                                  <SimpleGrid columns={{ base: 1, md: 3 }} spacing={4}>
                                    <Stat>
                                      <StatLabel>Total Tokens</StatLabel>
                                      <StatNumber>{analyticsResults.diversity?.total_tokens?.toLocaleString()}</StatNumber>
                                    </Stat>
                                    <Stat>
                                      <StatLabel>Vocabulary Size</StatLabel>
                                      <StatNumber>{analyticsResults.diversity?.vocabulary_size?.toLocaleString()}</StatNumber>
                                    </Stat>
                                    <Stat>
                                      <StatLabel>Overall Diversity</StatLabel>
                                      <StatNumber color={analyticsResults.diversity?.diversity_score > 0.5 ? 'green.500' : 'orange.500'}>
                                        {(analyticsResults.diversity?.diversity_score * 100).toFixed(1)}%
                                      </StatNumber>
                                    </Stat>
                                  </SimpleGrid>
                                </VStack>
                              </TabPanel>

                              {/* Text Stats Tab */}
                              <TabPanel>
                                <VStack spacing={4} align="stretch">
                                  <Text fontWeight="bold">Text Statistics</Text>
                                  
                                  <SimpleGrid columns={{ base: 1, md: 2 }} spacing={6}>
                                    <Box>
                                      <Text fontWeight="semibold" mb={2}>Question/User Message Length</Text>
                                      <Table variant="simple" size="sm">
                                        <Tbody>
                                          <Tr>
                                            <Td>Mean</Td>
                                            <Td isNumeric>{analyticsResults.text_stats?.question_length?.mean_length?.toFixed(1)} words</Td>
                                          </Tr>
                                          <Tr>
                                            <Td>Median</Td>
                                            <Td isNumeric>{analyticsResults.text_stats?.question_length?.median_length?.toFixed(1)} words</Td>
                                          </Tr>
                                          <Tr>
                                            <Td>Min / Max</Td>
                                            <Td isNumeric>
                                              {analyticsResults.text_stats?.question_length?.min_length} / {analyticsResults.text_stats?.question_length?.max_length}
                                            </Td>
                                          </Tr>
                                        </Tbody>
                                      </Table>
                                    </Box>

                                    <Box>
                                      <Text fontWeight="semibold" mb={2}>Answer/Assistant Message Length</Text>
                                      <Table variant="simple" size="sm">
                                        <Tbody>
                                          <Tr>
                                            <Td>Mean</Td>
                                            <Td isNumeric>{analyticsResults.text_stats?.answer_length?.mean_length?.toFixed(1)} words</Td>
                                          </Tr>
                                          <Tr>
                                            <Td>Median</Td>
                                            <Td isNumeric>{analyticsResults.text_stats?.answer_length?.median_length?.toFixed(1)} words</Td>
                                          </Tr>
                                          <Tr>
                                            <Td>Min / Max</Td>
                                            <Td isNumeric>
                                              {analyticsResults.text_stats?.answer_length?.min_length} / {analyticsResults.text_stats?.answer_length?.max_length}
                                            </Td>
                                          </Tr>
                                        </Tbody>
                                      </Table>
                                    </Box>
                                  </SimpleGrid>

                                  <Divider />

                                  {/* Length Distribution Chart */}
                                  <Text fontWeight="semibold" mb={2}>Answer Length Distribution</Text>
                                  {analyticsResults.text_stats?.answer_length?.histogram && (
                                    <Box height="250px">
                                      <ResponsiveContainer width="100%" height="100%">
                                        <BarChart
                                          data={Object.entries(analyticsResults.text_stats.answer_length.histogram).map(([range, count]) => ({
                                            range,
                                            count,
                                          }))}
                                          margin={{ top: 10, right: 30, left: 0, bottom: 5 }}
                                        >
                                          <CartesianGrid strokeDasharray="3 3" />
                                          <XAxis dataKey="range" tick={{ fontSize: 11 }} />
                                          <YAxis tick={{ fontSize: 11 }} />
                                          <RechartsTooltip />
                                          <Bar dataKey="count" fill="#3182CE" name="Examples" />
                                        </BarChart>
                                      </ResponsiveContainer>
                                    </Box>
                                  )}

                                  <Divider />

                                  <Text fontWeight="semibold">Content Patterns</Text>
                                  <SimpleGrid columns={{ base: 2, md: 4 }} spacing={4}>
                                    <Stat>
                                      <StatLabel>Q/A Ratio</StatLabel>
                                      <StatNumber>{analyticsResults.text_stats?.avg_qa_ratio?.toFixed(2)}x</StatNumber>
                                      <StatHelpText>Answer vs Question length</StatHelpText>
                                    </Stat>
                                    <Stat>
                                      <StatLabel>With Code</StatLabel>
                                      <StatNumber>{analyticsResults.text_stats?.has_code_blocks}</StatNumber>
                                    </Stat>
                                    <Stat>
                                      <StatLabel>With Lists</StatLabel>
                                      <StatNumber>{analyticsResults.text_stats?.has_lists}</StatNumber>
                                    </Stat>
                                    <Stat>
                                      <StatLabel>With URLs</StatLabel>
                                      <StatNumber>{analyticsResults.text_stats?.has_urls}</StatNumber>
                                    </Stat>
                                  </SimpleGrid>

                                  <Divider />

                                  <Text fontWeight="semibold">Quality Indicators</Text>
                                  <SimpleGrid columns={{ base: 1, md: 3 }} spacing={4}>
                                    <Stat>
                                      <StatLabel>Empty Responses</StatLabel>
                                      <StatNumber color={analyticsResults.text_stats?.empty_responses > 0 ? 'red.500' : 'green.500'}>
                                        {analyticsResults.text_stats?.empty_responses}
                                      </StatNumber>
                                    </Stat>
                                    <Stat>
                                      <StatLabel>Very Short (&lt;10 words)</StatLabel>
                                      <StatNumber color={analyticsResults.text_stats?.very_short_responses > analyticsResults.total_examples * 0.1 ? 'orange.500' : 'green.500'}>
                                        {analyticsResults.text_stats?.very_short_responses}
                                      </StatNumber>
                                    </Stat>
                                    <Stat>
                                      <StatLabel>Very Long (&gt;500 words)</StatLabel>
                                      <StatNumber>{analyticsResults.text_stats?.very_long_responses}</StatNumber>
                                    </Stat>
                                  </SimpleGrid>

                                  {analyticsResults.text_stats?.languages && Object.keys(analyticsResults.text_stats.languages).length > 0 && (
                                    <>
                                      <Divider />
                                      <Text fontWeight="semibold">Languages Detected</Text>
                                      <HStack spacing={2} wrap="wrap">
                                        {Object.entries(analyticsResults.text_stats.languages).map(([lang, count]) => (
                                          <Badge key={lang} colorScheme="blue" fontSize="sm">
                                            {lang}: {count}
                                          </Badge>
                                        ))}
                                      </HStack>
                                    </>
                                  )}
                                </VStack>
                              </TabPanel>

                              {/* Semantic Clusters Tab */}
                              <TabPanel>
                                <VStack spacing={4} align="stretch">
                                  <Text fontWeight="bold">Semantic Clustering</Text>
                                  
                                  <SimpleGrid columns={{ base: 1, md: 4 }} spacing={4}>
                                    <Stat>
                                      <StatLabel>Clusters Found</StatLabel>
                                      <StatNumber>{analyticsResults.semantic?.num_clusters}</StatNumber>
                                    </Stat>
                                    <Stat>
                                      <StatLabel>Semantic Diversity</StatLabel>
                                      <StatNumber color={analyticsResults.semantic?.semantic_diversity > 0.5 ? 'green.500' : 'orange.500'}>
                                        {(analyticsResults.semantic?.semantic_diversity * 100).toFixed(1)}%
                                      </StatNumber>
                                    </Stat>
                                    <Stat>
                                      <StatLabel>Avg Cluster Size</StatLabel>
                                      <StatNumber>{analyticsResults.semantic?.avg_cluster_size?.toFixed(1)}</StatNumber>
                                    </Stat>
                                    <Stat>
                                      <StatLabel>Largest Cluster</StatLabel>
                                      <StatNumber color={analyticsResults.semantic?.largest_cluster_ratio > 0.4 ? 'orange.500' : 'green.500'}>
                                        {(analyticsResults.semantic?.largest_cluster_ratio * 100).toFixed(1)}%
                                      </StatNumber>
                                      <StatHelpText>of total dataset</StatHelpText>
                                    </Stat>
                                  </SimpleGrid>

                                  {analyticsResults.semantic?.topic_distribution && Object.keys(analyticsResults.semantic.topic_distribution).length > 0 && (
                                    <>
                                      <Divider />
                                      <Text fontWeight="semibold">Topic Distribution</Text>
                                      
                                      {/* Pie Chart for Topics */}
                                      <SimpleGrid columns={{ base: 1, md: 2 }} spacing={4}>
                                        <Box height="280px">
                                          <ResponsiveContainer width="100%" height="100%">
                                            <PieChart>
                                              <Pie
                                                data={Object.entries(analyticsResults.semantic.topic_distribution)
                                                  .sort((a, b) => b[1] - a[1])
                                                  .slice(0, 8)
                                                  .map(([topic, pct], idx) => ({
                                                    name: topic.replace(/Cluster \d+: /, '').substring(0, 20),
                                                    value: pct,
                                                  }))}
                                                cx="50%"
                                                cy="50%"
                                                innerRadius={50}
                                                outerRadius={90}
                                                paddingAngle={2}
                                                dataKey="value"
                                                label={({ name, percent }) => `${(percent * 100).toFixed(0)}%`}
                                                labelLine={false}
                                              >
                                                {Object.entries(analyticsResults.semantic.topic_distribution)
                                                  .slice(0, 8)
                                                  .map((entry, index) => (
                                                    <Cell 
                                                      key={`cell-${index}`} 
                                                      fill={['#3182CE', '#38A169', '#DD6B20', '#805AD5', '#D53F8C', '#00B5D8', '#718096', '#E53E3E'][index % 8]} 
                                                    />
                                                  ))}
                                              </Pie>
                                              <RechartsTooltip formatter={(value) => `${value}%`} />
                                              <Legend 
                                                layout="vertical" 
                                                align="right" 
                                                verticalAlign="middle"
                                                formatter={(value) => value.substring(0, 15) + (value.length > 15 ? '...' : '')}
                                              />
                                            </PieChart>
                                          </ResponsiveContainer>
                                        </Box>

                                        <Table variant="simple" size="sm">
                                          <Thead>
                                            <Tr>
                                              <Th>Topic</Th>
                                              <Th isNumeric>%</Th>
                                            </Tr>
                                          </Thead>
                                          <Tbody>
                                            {Object.entries(analyticsResults.semantic.topic_distribution)
                                              .sort((a, b) => b[1] - a[1])
                                              .map(([topic, pct]) => (
                                                <Tr key={topic}>
                                                  <Td fontSize="sm">{topic}</Td>
                                                  <Td isNumeric>
                                                    <Badge colorScheme="blue">{pct}%</Badge>
                                                  </Td>
                                                </Tr>
                                              ))}
                                          </Tbody>
                                        </Table>
                                      </SimpleGrid>
                                    </>
                                  )}

                                  {analyticsResults.semantic?.clusters && analyticsResults.semantic.clusters.length > 0 && (
                                    <>
                                      <Divider />
                                      <Accordion allowToggle>
                                        <AccordionItem>
                                          <AccordionButton>
                                            <Box flex="1" textAlign="left">
                                              View Cluster Details
                                            </Box>
                                            <AccordionIcon />
                                          </AccordionButton>
                                          <AccordionPanel>
                                            <VStack spacing={3} align="stretch">
                                              {analyticsResults.semantic.clusters.map((cluster) => (
                                                <Box key={cluster.cluster_id} p={3} borderWidth="1px" borderRadius="md">
                                                  <HStack justify="space-between" mb={2}>
                                                    <Text fontWeight="bold">Cluster {cluster.cluster_id + 1}</Text>
                                                    <Badge colorScheme="blue">{cluster.size} examples</Badge>
                                                  </HStack>
                                                  <Text fontSize="sm" color="gray.500" mb={2}>
                                                    Keywords: {cluster.keywords?.join(', ')}
                                                  </Text>
                                                  <Text fontSize="xs" noOfLines={2}>
                                                    Representative: "{cluster.centroid_text}"
                                                  </Text>
                                                </Box>
                                              ))}
                                            </VStack>
                                          </AccordionPanel>
                                        </AccordionItem>
                                      </Accordion>
                                    </>
                                  )}
                                </VStack>
                              </TabPanel>

                              {/* Recommendations Tab */}
                              <TabPanel>
                                <VStack spacing={4} align="stretch">
                                  <Text fontWeight="bold">Recommendations</Text>
                                  
                                  {analyticsResults.recommendations && analyticsResults.recommendations.length > 0 ? (
                                    <VStack spacing={3} align="stretch">
                                      {analyticsResults.recommendations.map((rec, idx) => (
                                        <Alert key={idx} status={rec.includes('Ready') || rec.includes('good') ? 'success' : 'info'}>
                                          <AlertIcon />
                                          {rec}
                                        </Alert>
                                      ))}
                                    </VStack>
                                  ) : (
                                    <Alert status="success">
                                      <AlertIcon />
                                      No specific recommendations. Dataset looks good!
                                    </Alert>
                                  )}
                                </VStack>
                              </TabPanel>
                            </TabPanels>
                          </Tabs>
                        </CardBody>
                      </Card>
                    </>
                  )}
                </VStack>
              </TabPanel>

              {/* Safety & Compliance Tab */}
              <TabPanel>
                <VStack spacing={6} align="stretch">
                  <Alert status="info">
                    <AlertIcon />
                    <Box>
                      <AlertTitle>Safety & Compliance Checks</AlertTitle>
                      <AlertDescription>
                        Detoxify (toxicity), Microsoft Presidio (PII), and a MinHash-LSH deduplicator run locally—no API keys, fully private, and powered by pre-trained open-source models.
                      </AlertDescription>
                    </Box>
                  </Alert>

                  <Button
                    colorScheme="blue"
                    onClick={handleSafetyAnalysis}
                    isDisabled={!selectedDataset}
                    isLoading={isAnalyzing}
                    width="250px"
                  >
                    Run Safety Analysis
                  </Button>

                  {isAnalyzing && (
                    <Card bg={cardBg} borderWidth="1px" borderColor={borderColor}>
                      <CardBody>
                        <VStack spacing={4}>
                          <Spinner size="xl" color="blue.500" />
                          <Text>Analyzing dataset safety...</Text>
                          <Text fontSize="sm" color="gray.500">
                            This may take a few minutes for large datasets
                          </Text>
                        </VStack>
                      </CardBody>
                    </Card>
                  )}

                  {analysisResults && (
                    <>
                      <Card bg={cardBg} borderWidth="1px" borderColor={borderColor}>
                        <CardHeader>
                          <Heading size="sm">Safety Summary</Heading>
                        </CardHeader>
                        <CardBody>
                          <SimpleGrid columns={{ base: 1, md: 4 }} spacing={6}>
                            <Stat>
                              <StatLabel>Total Examples</StatLabel>
                              <StatNumber>{analysisResults.summary.total_examples}</StatNumber>
                            </Stat>

                            <Stat>
                              <StatLabel>Toxic Content</StatLabel>
                              <StatNumber color={analysisResults.summary.toxic_percentage > 10 ? 'red.500' : 'green.500'}>
                                {analysisResults.summary.toxic_count}
                              </StatNumber>
                              <StatHelpText>
                                {analysisResults.summary.toxic_percentage}% of dataset
                              </StatHelpText>
                            </Stat>

                            <Stat>
                              <StatLabel>PII Detected</StatLabel>
                              <StatNumber color={analysisResults.summary.pii_percentage > 5 ? 'orange.500' : 'green.500'}>
                                {analysisResults.summary.pii_count}
                              </StatNumber>
                              <StatHelpText>
                                {analysisResults.summary.pii_percentage}% of dataset
                              </StatHelpText>
                            </Stat>

                            <Stat>
                              <StatLabel>Duplicate Groups</StatLabel>
                              <StatNumber color={analysisResults.summary.duplicate_percentage > 20 ? 'yellow.500' : 'green.500'}>
                                {analysisResults.summary.duplicate_groups}
                              </StatNumber>
                              <StatHelpText>
                                {analysisResults.summary.duplicate_percentage}% duplicates
                              </StatHelpText>
                            </Stat>
                          </SimpleGrid>

                          <VStack spacing={3} mt={6} align="stretch">
                            {analysisResults.summary.toxic_percentage > 10 && (
                              <Alert status="error">
                                <AlertIcon />
                                <AlertTitle>High Toxicity Detected!</AlertTitle>
                                <AlertDescription>
                                  {analysisResults.summary.toxic_percentage}% of your dataset contains toxic content.
                                </AlertDescription>
                              </Alert>
                            )}

                            {analysisResults.summary.pii_percentage > 5 && (
                              <Alert status="warning">
                                <AlertIcon />
                                <AlertTitle>PII Found!</AlertTitle>
                                <AlertDescription>
                                  {analysisResults.summary.pii_count} examples contain personally identifiable information.
                                  <Button ml={4} size="sm" colorScheme="orange" onClick={handleAnonymization}>
                                    Anonymize PII
                                  </Button>
                                </AlertDescription>
                              </Alert>
                            )}

                            {analysisResults.summary.duplicate_percentage > 20 && (
                              <Alert status="info">
                                <AlertIcon />
                                <AlertTitle>High Duplication Rate</AlertTitle>
                                <AlertDescription>
                                  {analysisResults.summary.duplicate_percentage}% of your dataset contains duplicates.
                                </AlertDescription>
                              </Alert>
                            )}

                            {analysisResults.summary.toxic_percentage <= 10 && 
                             analysisResults.summary.pii_percentage <= 5 && 
                             analysisResults.summary.duplicate_percentage <= 20 && (
                              <Alert status="success">
                                <AlertIcon />
                                <AlertTitle>Excellent Quality!</AlertTitle>
                                <AlertDescription>
                                  Your dataset passes all safety checks.
                                </AlertDescription>
                              </Alert>
                            )}
                          </VStack>
                        </CardBody>
                      </Card>

                      {/* Detailed Results */}
                      <Card bg={cardBg} borderWidth="1px" borderColor={borderColor}>
                        <CardBody>
                          <Tabs variant="line" colorScheme="blue">
                            <TabList mb={4}>
                              <Tab>Toxicity</Tab>
                              <Tab>PII</Tab>
                              <Tab>Duplicates</Tab>
                            </TabList>

                            <TabPanels>
                              {/* Toxicity Details */}
                              <TabPanel>
                                <VStack spacing={4} align="stretch">
                                  <Text fontWeight="bold">Toxicity Detection Results</Text>
                                  
                                  <SimpleGrid columns={{ base: 1, md: 3 }} spacing={4}>
                                    <Stat>
                                      <StatLabel>Analyzed</StatLabel>
                                      <StatNumber>{analysisResults.toxicity.total_analyzed}</StatNumber>
                                    </Stat>
                                    <Stat>
                                      <StatLabel>Toxic Examples</StatLabel>
                                      <StatNumber color="red.500">{analysisResults.toxicity.toxic_count}</StatNumber>
                                    </Stat>
                                    <Stat>
                                      <StatLabel>Toxicity Rate</StatLabel>
                                      <StatNumber>{analysisResults.toxicity.toxic_percentage}%</StatNumber>
                                    </Stat>
                                  </SimpleGrid>

                                  {analysisResults.toxicity.results && analysisResults.toxicity.results.length > 0 && (
                                    <Accordion allowToggle>
                                      <AccordionItem>
                                        <AccordionButton>
                                          <Box flex="1" textAlign="left">
                                            View Toxic Examples (First 10)
                                          </Box>
                                          <AccordionIcon />
                                        </AccordionButton>
                                        <AccordionPanel>
                                          <Table variant="simple" size="sm">
                                            <Thead>
                                              <Tr>
                                                <Th>Status</Th>
                                                <Th>Toxicity</Th>
                                                <Th>Severe</Th>
                                                <Th>Obscene</Th>
                                                <Th>Threat</Th>
                                                <Th>Insult</Th>
                                              </Tr>
                                            </Thead>
                                            <Tbody>
                                              {analysisResults.toxicity.results.slice(0, 10).filter(r => r.is_toxic).map((result, idx) => (
                                                <Tr key={idx}>
                                                  <Td>
                                                    <Badge colorScheme={result.is_toxic ? 'red' : 'green'}>
                                                      {result.is_toxic ? 'Toxic' : 'Clean'}
                                                    </Badge>
                                                  </Td>
                                                  <Td>{(result.scores.toxicity * 100).toFixed(1)}%</Td>
                                                  <Td>{(result.scores.severe_toxicity * 100).toFixed(1)}%</Td>
                                                  <Td>{(result.scores.obscene * 100).toFixed(1)}%</Td>
                                                  <Td>{(result.scores.threat * 100).toFixed(1)}%</Td>
                                                  <Td>{(result.scores.insult * 100).toFixed(1)}%</Td>
                                                </Tr>
                                              ))}
                                            </Tbody>
                                          </Table>
                                        </AccordionPanel>
                                      </AccordionItem>
                                    </Accordion>
                                  )}
                                </VStack>
                              </TabPanel>

                              {/* PII Details */}
                              <TabPanel>
                                <VStack spacing={4} align="stretch">
                                  <Text fontWeight="bold">PII Detection Results</Text>
                                  
                                  <SimpleGrid columns={{ base: 1, md: 3 }} spacing={4}>
                                    <Stat>
                                      <StatLabel>Analyzed</StatLabel>
                                      <StatNumber>{analysisResults.pii.total_analyzed}</StatNumber>
                                    </Stat>
                                    <Stat>
                                      <StatLabel>Examples with PII</StatLabel>
                                      <StatNumber color="orange.500">{analysisResults.pii.pii_count}</StatNumber>
                                    </Stat>
                                    <Stat>
                                      <StatLabel>PII Rate</StatLabel>
                                      <StatNumber>{analysisResults.pii.pii_percentage}%</StatNumber>
                                    </Stat>
                                  </SimpleGrid>

                                  {analysisResults.pii.pii_types && Object.keys(analysisResults.pii.pii_types).length > 0 && (
                                    <>
                                      <Text fontWeight="bold" mt={4}>PII Types Detected:</Text>
                                      <Table variant="simple" size="sm">
                                        <Thead>
                                          <Tr>
                                            <Th>Entity Type</Th>
                                            <Th isNumeric>Count</Th>
                                          </Tr>
                                        </Thead>
                                        <Tbody>
                                          {Object.entries(analysisResults.pii.pii_types).map(([type, count]) => (
                                            <Tr key={type}>
                                              <Td>{type}</Td>
                                              <Td isNumeric>{count}</Td>
                                            </Tr>
                                          ))}
                                        </Tbody>
                                      </Table>
                                    </>
                                  )}

                                  <Button colorScheme="orange" onClick={handleAnonymization} mt={4} width="200px">
                                    Anonymize All PII
                                  </Button>
                                </VStack>
                              </TabPanel>

                              {/* Duplicates Details */}
                              <TabPanel>
                                <VStack spacing={4} align="stretch">
                                  <Text fontWeight="bold">Duplicate Detection Results</Text>
                                  
                                  <SimpleGrid columns={{ base: 1, md: 3 }} spacing={4}>
                                    <Stat>
                                      <StatLabel>Analyzed</StatLabel>
                                      <StatNumber>{analysisResults.duplicates.total_analyzed}</StatNumber>
                                    </Stat>
                                    <Stat>
                                      <StatLabel>Duplicate Groups</StatLabel>
                                      <StatNumber color="yellow.500">{analysisResults.duplicates.duplicate_groups}</StatNumber>
                                    </Stat>
                                    <Stat>
                                      <StatLabel>Total Duplicates</StatLabel>
                                      <StatNumber>{analysisResults.duplicates.total_duplicates}</StatNumber>
                                    </Stat>
                                  </SimpleGrid>

                                  {analysisResults.duplicates.groups && analysisResults.duplicates.groups.length > 0 && (
                                    <Accordion allowToggle>
                                      <AccordionItem>
                                        <AccordionButton>
                                          <Box flex="1" textAlign="left">
                                            View Duplicate Groups (First 10)
                                          </Box>
                                          <AccordionIcon />
                                        </AccordionButton>
                                        <AccordionPanel>
                                          <VStack spacing={3} align="stretch">
                                            {analysisResults.duplicates.groups.slice(0, 10).map((group, idx) => (
                                              <Box key={idx} p={3} borderWidth="1px" borderRadius="md">
                                                <Text fontWeight="bold">Group {idx + 1}:</Text>
                                                <Text fontSize="sm" color="gray.500">
                                                  {group.group.length} similar examples found
                                                </Text>
                                              </Box>
                                            ))}
                                          </VStack>
                                        </AccordionPanel>
                                      </AccordionItem>
                                    </Accordion>
                                  )}
                                </VStack>
                              </TabPanel>
                            </TabPanels>
                          </Tabs>
                        </CardBody>
                      </Card>
                    </>
                  )}
                </VStack>
              </TabPanel>
            </TabPanels>
          </Tabs>
        </CardBody>
      </Card>
    </Box>
  );
}

export default Quality;
