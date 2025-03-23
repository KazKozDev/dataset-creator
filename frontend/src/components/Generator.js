import React, { useState, useEffect } from 'react';
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
  Stack,
  Text,
  Textarea,
  useToast,
  Alert,
  AlertIcon,
  Progress,
  Badge,
  Code,
  Flex,
  SimpleGrid,
  VStack,
  HStack,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  NumberIncrementStepper,
  NumberDecrementStepper,
  Slider,
  SliderTrack,
  SliderFilledTrack,
  SliderThumb,
  useColorModeValue,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Switch,
  Divider,
  Radio,
  RadioGroup,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText
} from '@chakra-ui/react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { getDomains, getModels, startGenerator, getGeneratorStatus } from '../services/api';
import DomainCard from './common/DomainCard';

const Generator = () => {
  const toast = useToast();
  const [selectedDomain, setSelectedDomain] = useState(null);
  const [selectedSubdomain, setSelectedSubdomain] = useState(null);
  const [generationParams, setGenerationParams] = useState({
    format: 'chat',
    language: 'en',
    count: 50,
    temperature: 0.7,
    provider: 'ollama',
    model: '',
  });
  const [currentStep, setCurrentStep] = useState(0);
  const [jobId, setJobId] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);
  
  const cardBg = useColorModeValue('white', 'gray.700');
  const borderColor = useColorModeValue('gray.200', 'gray.600');
  
  // Fetch domains
  const { data: domainsData, isLoading: isLoadingDomains } = useQuery({
    queryKey: ['domains'],
    queryFn: getDomains
  });

  // Fetch models for selected provider
  const { data: modelsData, isLoading: isLoadingModels } = useQuery({
    queryKey: ['models', generationParams.provider],
    queryFn: () => getModels(generationParams.provider),
    enabled: !!generationParams.provider
  });
  
  // Fetch job status
  useEffect(() => {
    let interval;
    if (jobId) {
      interval = setInterval(async () => {
        try {
          const status = await getGeneratorStatus(jobId);
          setJobStatus(status);
          
          if (['completed', 'failed', 'cancelled'].includes(status.status)) {
            clearInterval(interval);
          }
        } catch (error) {
          console.error('Error fetching job status:', error);
          clearInterval(interval);
        }
      }, 2000);
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [jobId]);
  
  // Start generation mutation
  const startGenerationMutation = useMutation({
    mutationFn: startGenerator,
    onSuccess: (data) => {
      setJobId(data.job_id);
      setCurrentStep(2);
      toast({
        title: 'Generation started',
        description: `Job ID: ${data.job_id}`,
        status: 'success',
        duration: 5000,
        isClosable: true,
      });
    },
    onError: (error) => {
      toast({
        title: 'Error starting generation',
        description: error.response?.data?.detail || 'Unknown error',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    },
  });
  
  const handleDomainSelect = (domain) => {
    setSelectedDomain(domain);
    setSelectedSubdomain(null);
    setCurrentStep(1);
  };
  
  const handleSubdomainSelect = (subdomain) => {
    setSelectedSubdomain(subdomain);
  };
  
  const handleParamChange = (key, value) => {
    setGenerationParams({
      ...generationParams,
      [key]: value,
      // Reset model when provider changes
      ...(key === 'provider' ? { model: '' } : {})
    });
  };
  
  const handleStartGeneration = () => {
    const params = {
      domain: selectedDomain.key,
      subdomain: selectedSubdomain?.key,
      ...generationParams,
    };
    
    startGenerationMutation.mutate(params);
  };
  
  const handleReset = () => {
    setSelectedDomain(null);
    setSelectedSubdomain(null);
    setJobId(null);
    setJobStatus(null);
    setCurrentStep(0);
  };
  
  // Calculate generation progress
  const calculateProgress = () => {
    if (!jobStatus) return 0;
    return (jobStatus.examples_generated / jobStatus.examples_requested) * 100;
  };
  
  // Render domain selection step
  const renderDomainSelection = () => {
    if (isLoadingDomains) {
      return (
        <Box textAlign="center" py={10}>
          <Text>Loading domains...</Text>
        </Box>
      );
    }
    
    return (
      <Box>
        <Text fontSize="lg" mb={4}>
          Select a domain for your dataset:
        </Text>
        
        <SimpleGrid columns={{ base: 1, md: 2, lg: 3 }} spacing={6}>
          {domainsData?.domains.map((domain) => (
            <DomainCard
              key={domain.key}
              domain={domain}
              onClick={() => handleDomainSelect(domain)}
            />
          ))}
        </SimpleGrid>
      </Box>
    );
  };
  
  // Render subdomain selection step
  const renderSubdomainSelection = () => {
    if (!selectedDomain) return null;
    
    return (
      <Box>
        <HStack mb={4}>
          <Button size="sm" onClick={() => setCurrentStep(0)}>
            Back
          </Button>
          <Text fontSize="lg">
            {selectedDomain.name} &gt; Select a subdomain
          </Text>
        </HStack>
        
        <SimpleGrid columns={{ base: 1, md: 2 }} spacing={4}>
          {Object.entries(selectedDomain.subdomains).map(([key, subdomain]) => (
            <Card
              key={key}
              cursor="pointer"
              bg={selectedSubdomain?.key === key ? 'blue.50' : cardBg}
              borderWidth="1px"
              borderColor={selectedSubdomain?.key === key ? 'blue.500' : borderColor}
              borderRadius="lg"
              overflow="hidden"
              onClick={() => handleSubdomainSelect({ key, ...subdomain })}
              _hover={{ borderColor: 'blue.300' }}
            >
              <CardHeader>
                <Heading size="lg">{subdomain.name}</Heading>
              </CardHeader>
              <CardBody>
                <Text>{subdomain.description}</Text>
                <Box mt={2}>
                  {subdomain.scenarios.map((scenario) => (
                    <Badge key={scenario} mr={1} mb={1} colorScheme="blue">
                      {scenario}
                    </Badge>
                  ))}
                </Box>
              </CardBody>
            </Card>
          ))}
        </SimpleGrid>
        
        <Divider my={4} />
        
        <Box>
          <Heading size="lg" mb={4}>Generation Parameters</Heading>
          
          <SimpleGrid columns={{ base: 1, md: 2 }} spacing={6}>
            <FormControl>
              <FormLabel>Output Format</FormLabel>
              <RadioGroup
                value={generationParams.format}
                onChange={(value) => handleParamChange('format', value)}
              >
                <Stack direction="row">
                  <Radio value="chat">Chat</Radio>
                  <Radio value="instruction">Instruction</Radio>
                </Stack>
              </RadioGroup>
            </FormControl>
            
            <FormControl>
              <FormLabel>Language</FormLabel>
              <Select
                value={generationParams.language}
                onChange={(e) => handleParamChange('language', e.target.value)}
              >
                <option value="en">English</option>
                <option value="ru">Russian</option>
              </Select>
            </FormControl>
            
            <FormControl>
              <FormLabel>Number of Examples</FormLabel>
              <NumberInput
                min={1}
                max={500}
                value={generationParams.count}
                onChange={(valueString) => handleParamChange('count', parseInt(valueString))}
              >
                <NumberInputField />
                <NumberInputStepper>
                  <NumberIncrementStepper />
                  <NumberDecrementStepper />
                </NumberInputStepper>
              </NumberInput>
            </FormControl>
            
            <FormControl>
              <FormLabel>Temperature: {generationParams.temperature}</FormLabel>
              <Slider
                min={0.1}
                max={1.0}
                step={0.1}
                value={generationParams.temperature}
                onChange={(value) => handleParamChange('temperature', value)}
              >
                <SliderTrack>
                  <SliderFilledTrack />
                </SliderTrack>
                <SliderThumb boxSize={6} />
              </Slider>
            </FormControl>
            
            <FormControl>
              <FormLabel>Provider</FormLabel>
              <Select
                value={generationParams.provider}
                onChange={(e) => handleParamChange('provider', e.target.value)}
              >
                <option value="ollama">Ollama</option>
                <option value="openai">OpenAI</option>
                <option value="anthropic">Anthropic</option>
              </Select>
            </FormControl>
            
            <FormControl>
              <FormLabel>Model</FormLabel>
              <Select
                value={generationParams.model}
                onChange={(e) => handleParamChange('model', e.target.value)}
                placeholder="Select a model"
                isDisabled={isLoadingModels}
              >
                {isLoadingModels ? (
                  <option>Loading models...</option>
                ) : generationParams.provider === 'openai' ? (
                  <>
                    <option value="gpt-4o">GPT-4o</option>
                    <option value="03-mini-high">03-mini-high</option>
                  </>
                ) : generationParams.provider === 'anthropic' ? (
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
            </FormControl>
          </SimpleGrid>
          
          <Button
            mt={8}
            colorScheme="blue"
            isDisabled={!selectedSubdomain}
            onClick={handleStartGeneration}
            isLoading={startGenerationMutation.isLoading}
          >
            Start Generation
          </Button>
        </Box>
      </Box>
    );
  };
  
  // Render generation progress step
  const renderGenerationProgress = () => {
    if (!jobId) return null;
    
    return (
      <Box>
        <HStack mb={4}>
          <Button size="sm" onClick={handleReset}>
            Start New Generation
          </Button>
          <Text fontSize="lg">Generation in Progress</Text>
        </HStack>
        
        <Card p={6}>
          <VStack spacing={4} align="stretch">
            <Heading size="lg">
              Generating {selectedDomain.name} &gt; {selectedSubdomain.name} Dataset
            </Heading>
            
            <Text>
              Format: {generationParams.format === 'chat' ? 'Chat Format' : 'Instruction Format'}
            </Text>
            
            <Text>
              Language: {generationParams.language === 'en' ? 'English' : 'Russian'}
            </Text>
            
            <Text>Job ID: {jobId}</Text>
            
            <Text>
              Status: <Badge colorScheme={jobStatus?.status === 'completed' ? 'green' : jobStatus?.status === 'failed' ? 'red' : 'blue'}>
                {jobStatus?.status || 'Pending'}
              </Badge>
            </Text>
            
            <Text>
              Progress: {jobStatus?.examples_generated || 0} / {jobStatus?.examples_requested || 0} examples
            </Text>
            
            <Progress
              value={calculateProgress()}
              size="lg"
              colorScheme="blue"
              borderRadius="md"
            />
            
            {jobStatus?.status === 'completed' && jobStatus?.dataset_id && (
              <Button
                colorScheme="green"
                as="a"
                href={`/datasets/${jobStatus.dataset_id}`}
              >
                View Generated Dataset
              </Button>
            )}
            
            {jobStatus?.status === 'failed' && (
              <Box>
                <Text color="red.500">Generation failed:</Text>
                <Text>{jobStatus?.errors?.join(', ') || 'Unknown error'}</Text>
              </Box>
            )}
          </VStack>
        </Card>
      </Box>
    );
  };
  
  // Render the current step
  const renderCurrentStep = () => {
    switch (currentStep) {
      case 0:
        return renderDomainSelection();
      case 1:
        return renderSubdomainSelection();
      case 2:
        return renderGenerationProgress();
      default:
        return null;
    }
  };
  
  return (
    <Box>
      <Heading size="lg" mb={6}>Dataset Generator</Heading>
      
      {renderCurrentStep()}
    </Box>
  );
};

export default Generator;